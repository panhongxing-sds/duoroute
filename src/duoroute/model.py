"""DuoRoute dual-channel router with model-card embeddings."""

from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, *, hidden_dim: int | None = None):
        super().__init__()
        hidden = hidden_dim or max(out_dim, in_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DuoRouteModel(nn.Module):
    """
    Channel A: query + model-card embedding -> reward
    Channel B: query + response + model-card embedding -> reward
    """

    def __init__(
        self,
        query_embeddings: torch.Tensor,
        model_embeddings: torch.Tensor,
        *,
        hidden_dim: int = 64,
        query_dim: int | None = None,
        response_dim: int = 256,
        model_dim: int | None = None,
        use_id_fallback: bool = False,
        num_models: int | None = None,
    ):
        super().__init__()
        qdim = query_dim or query_embeddings.shape[1]
        mdim = model_dim or model_embeddings.shape[1]
        self.use_id_fallback = use_id_fallback

        self.query_table = nn.Embedding(query_embeddings.shape[0], qdim)
        self.query_table.weight = nn.Parameter(query_embeddings.clone(), requires_grad=False)
        self.model_card_emb = nn.Parameter(model_embeddings.clone(), requires_grad=False)

        self.query_projector = MLP(qdim, hidden_dim)
        self.model_projector = MLP(mdim, hidden_dim)
        self.response_projector = MLP(response_dim, hidden_dim)
        self.channel_a = MLP(hidden_dim * 2, 1)
        self.channel_b = MLP(hidden_dim * 3, 1)

        if use_id_fallback and num_models is not None:
            self.model_id_emb = nn.Embedding(num_models, hidden_dim)
        else:
            self.model_id_emb = None

    def _model_hidden(self, batch_size: int) -> torch.Tensor:
        model_h = self.model_projector(self.model_card_emb)
        if self.model_id_emb is not None:
            ids = torch.arange(model_h.size(0), device=model_h.device)
            model_h = model_h + self.model_id_emb(ids)
        return model_h.unsqueeze(0).expand(batch_size, -1, -1)

    def forward(
        self,
        prompt_ids: torch.Tensor,
        response_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, k, _ = response_emb.shape
        query_h = self.query_projector(self.query_table(prompt_ids))
        model_h = self._model_hidden(batch_size)

        query_exp = query_h.unsqueeze(1).expand(-1, k, -1)
        a_input = torch.cat([query_exp, model_h], dim=-1)
        pred_a = self.channel_a(a_input).squeeze(-1)

        response_h = self.response_projector(response_emb)
        b_input = torch.cat([query_exp, response_h, model_h], dim=-1)
        pred_b = self.channel_b(b_input).squeeze(-1)
        return pred_a, pred_b

    def forward_a(self, prompt_ids: torch.Tensor) -> torch.Tensor:
        batch_size = prompt_ids.size(0)
        k = self.model_card_emb.size(0)
        query_h = self.query_projector(self.query_table(prompt_ids))
        model_h = self._model_hidden(batch_size)
        query_exp = query_h.unsqueeze(1).expand(-1, k, -1)
        return self.channel_a(torch.cat([query_exp, model_h], dim=-1)).squeeze(-1)

    def forward_b(self, prompt_ids: torch.Tensor, response_emb: torch.Tensor) -> torch.Tensor:
        _, pred_b = self.forward(prompt_ids, response_emb)
        return pred_b

    def with_model_embeddings(self, model_embeddings: torch.Tensor) -> "DuoRouteModel":
        """Return a copy view for zero-shot unseen model cards at inference time."""
        clone = DuoRouteModel(
            self.query_table.weight.detach(),
            model_embeddings,
            hidden_dim=self.query_projector.net[0].out_features,
            query_dim=self.query_table.weight.shape[1],
            response_dim=self.response_projector.net[0].in_features,
            model_dim=model_embeddings.shape[1],
            use_id_fallback=self.use_id_fallback,
            num_models=model_embeddings.shape[0] if self.model_id_emb is not None else None,
        )
        clone.load_state_dict(self.state_dict(), strict=False)
        clone.model_card_emb = nn.Parameter(model_embeddings.clone(), requires_grad=False)
        return clone

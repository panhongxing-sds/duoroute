"""Query-conditioned full-response VAE (V1) and residual behavior VAE/AE (V1.1)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class QueryToResponseLinear(nn.Module):
    """P(x) = W x + b — query main effect on response embedding."""

    def __init__(self, query_dim: int, response_dim: int):
        super().__init__()
        self.linear = nn.Linear(query_dim, response_dim)

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        return self.linear(query)


class QueryConditionedResponseVAE(nn.Module):
    """V1: reconstruct full response s with query in decoder (baseline for comparison)."""

    def __init__(
        self,
        embed_dim: int,
        *,
        latent_dim: int = 64,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        enc_in = embed_dim * 2
        self.encoder = nn.Sequential(
            nn.Linear(enc_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)
        dec_in = latent_dim + embed_dim
        self.decoder = nn.Sequential(
            nn.Linear(dec_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def encode(self, response: torch.Tensor, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(torch.cat([response, query], dim=-1))
        return self.mu_head(h), self.logvar_head(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat([z, query], dim=-1))

    def forward(
        self, response: torch.Tensor, query: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(response, query)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, query), mu, logvar, z

    @torch.no_grad()
    def encode_latent(self, response: torch.Tensor, query: torch.Tensor, *, use_mu: bool = True) -> torch.Tensor:
        mu, logvar = self.encode(response, query)
        if use_mu:
            return mu
        return self.reparameterize(mu, logvar)


class ResidualBehaviorVAE(nn.Module):
    """
    V1.1: Encoder(r, x) -> z; Decoder(z) -> r_hat (no query in decoder).
    kl_weight=0 in training => deterministic AE (use mu as z).
    """

    def __init__(
        self,
        residual_dim: int,
        query_dim: int,
        *,
        latent_dim: int = 32,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.residual_dim = residual_dim
        self.query_dim = query_dim
        self.latent_dim = latent_dim
        enc_in = residual_dim + query_dim
        self.encoder = nn.Sequential(
            nn.Linear(enc_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, residual_dim),
        )

    def encode(self, residual: torch.Tensor, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(torch.cat([residual, query], dim=-1))
        return self.mu_head(h), self.logvar_head(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(
        self, residual: torch.Tensor, query: torch.Tensor, *, sample: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(residual, query)
        z = self.reparameterize(mu, logvar) if sample else mu
        return self.decode(z), mu, logvar, z

    @torch.no_grad()
    def encode_latent(
        self, residual: torch.Tensor, query: torch.Tensor, *, use_mu: bool = True
    ) -> torch.Tensor:
        mu, logvar = self.encode(residual, query)
        if use_mu:
            return mu
        return self.reparameterize(mu, logvar)


def vae_loss(
    target: torch.Tensor,
    recon: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    *,
    kl_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    recon_loss = F.mse_loss(recon, target)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon_loss + kl_weight * kl
    return total, {
        "recon": float(recon_loss.detach()),
        "kl": float(kl.detach()),
        "total": float(total.detach()),
    }


@torch.no_grad()
def normalize_residual(residual: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return F.normalize(residual, dim=-1, eps=eps)

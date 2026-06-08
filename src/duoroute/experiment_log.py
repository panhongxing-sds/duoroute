"""Structured JSON experiment records for reproducibility and review."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from duoroute.utils import project_root, save_json


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root(),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _command_line() -> str:
    return " ".join(sys.argv)


@dataclass
class ExperimentRecord:
    """Full experiment payload; always write to ``experiment.json`` + global index."""

    experiment_type: str
    run_id: str
    status: str = "running"
    started_at: str = field(default_factory=_utc_now)
    finished_at: Optional[str] = None
    pool: Optional[str] = None
    data_dir: Optional[str] = None
    config_path: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    command: str = field(default_factory=_command_line)
    git_commit: Optional[str] = field(default_factory=_git_commit)
    hostname: str = field(default_factory=socket.gethostname)
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def make_run_id(experiment_type: str, pool: str | None = None, tag: str | None = None) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [ts, experiment_type]
    if pool:
        parts.append(pool)
    if tag:
        parts.append(tag)
    return "_".join(parts)


def append_index(record: Dict[str, Any], *, index_path: Path | None = None) -> Path:
    index_path = index_path or (project_root() / "outputs/experiments/index.jsonl")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return index_path


def save_experiment(
    record: ExperimentRecord | Dict[str, Any],
    *,
    output_dir: Path | str,
    also_index: bool = True,
) -> Path:
    """Write ``experiment.json`` under *output_dir* and append one line to global index."""
    payload = record.to_dict() if isinstance(record, ExperimentRecord) else dict(record)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "experiment.json"
    save_json(path, payload)
    if also_index:
        append_index(payload)
    return path


def finish_experiment(
    record: ExperimentRecord,
    *,
    status: str,
    metrics: Optional[Dict[str, Any]] = None,
    artifacts: Optional[Dict[str, str]] = None,
    extra: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> ExperimentRecord:
    record.status = status
    record.finished_at = _utc_now()
    if metrics is not None:
        record.metrics.update(metrics)
    if artifacts is not None:
        record.artifacts.update(artifacts)
    if extra is not None:
        record.extra.update(extra)
    if error is not None:
        record.error = error
    return record

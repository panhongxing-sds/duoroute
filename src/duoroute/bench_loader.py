"""Load LLMRouterBench-style JSON results without external baseline dependencies."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import yaml
from loguru import logger

from duoroute.schema import BenchRecord


def _default_config() -> Dict[str, Any]:
    return {
        "results_dir": "results/bench",
        "filters": {
            "skip_demo": True,
            "datasets": None,
            "models": None,
            "splits": None,
            "exclude_datasets": None,
            "exclude_models": None,
            "exclude_splits": None,
        },
    }


def load_config(config_path: str | Path) -> Dict[str, Any]:
    with open(config_path, encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    if "baseline" in loaded:
        return loaded["baseline"]
    return loaded


class BenchLoader:
    """Minimal loader for `results/bench/<dataset>/<split>/<model>/*.json`."""

    def __init__(
        self,
        *,
        config_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        results_dir: Optional[str] = None,
    ):
        if config_path:
            self.config = load_config(config_path)
        elif config:
            self.config = config
        else:
            self.config = _default_config()

        if results_dir is not None:
            self.config["results_dir"] = results_dir

        self.results_dir = Path(self.config.get("results_dir", "results/bench"))
        self.filters = self.config.get("filters", {})
        logger.info(f"BenchLoader results_dir={self.results_dir}")

    def _should_skip_file(self, file_path: Path) -> bool:
        try:
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)
            if self.filters.get("skip_demo", True) and data.get("demo", False):
                return True
            if self.filters.get("datasets") is not None:
                if data.get("dataset_name") not in self.filters["datasets"]:
                    return True
            elif self.filters.get("exclude_datasets"):
                if data.get("dataset_name") in self.filters["exclude_datasets"]:
                    return True
            if self.filters.get("models") is not None:
                if data.get("model_name") not in self.filters["models"]:
                    return True
            elif self.filters.get("exclude_models"):
                if data.get("model_name") in self.filters["exclude_models"]:
                    return True
            if self.filters.get("splits") is not None:
                if data.get("split") not in self.filters["splits"]:
                    return True
            elif self.filters.get("exclude_splits"):
                if data.get("split") in self.filters["exclude_splits"]:
                    return True
            return False
        except Exception as exc:
            logger.warning(f"Skip {file_path}: {exc}")
            return True

    @staticmethod
    def _latest_file(files: List[Path]) -> Path:
        def ts(path: Path) -> int:
            match = re.search(r"(\d{8})_(\d{6})\.json$", path.name)
            if match:
                return int(match.group(1) + match.group(2))
            return int(path.stat().st_mtime)

        return max(files, key=ts)

    def _find_result_files(self) -> List[Path]:
        if not self.results_dir.exists():
            raise FileNotFoundError(f"Bench directory not found: {self.results_dir}")

        latest: Dict[tuple[str, str, str], Path] = {}
        for file_path in self.results_dir.rglob("*.json"):
            if self._should_skip_file(file_path):
                continue
            parts = file_path.relative_to(self.results_dir).parts
            if len(parts) < 4:
                continue
            dataset_id, split, model_name = parts[0], parts[1], parts[2]
            key = (dataset_id, split, model_name)
            if key not in latest or self._latest_file([latest[key], file_path]) == file_path:
                latest[key] = file_path
        files = sorted(latest.values())
        logger.info(f"Found {len(files)} result files")
        return files

    def iter_records(self) -> Iterator[BenchRecord]:
        total = 0
        for file_path in self._find_result_files():
            try:
                with open(file_path, encoding="utf-8") as f:
                    payload = json.load(f)
                dataset_id = payload.get("dataset_name", "")
                split = payload.get("split", "")
                model_name = payload.get("model_name", "")
                records = payload.get("records", [])
                for record_data in records:
                    def norm(value: Any) -> str:
                        if value is None:
                            return ""
                        if isinstance(value, (list, dict)):
                            return json.dumps(value, ensure_ascii=False)
                        return str(value)

                    yield BenchRecord(
                        dataset_id=dataset_id,
                        split=split,
                        model_name=model_name,
                        record_index=int(record_data.get("index", 0)),
                        origin_query=norm(record_data.get("origin_query", "")),
                        prompt=norm(record_data.get("prompt", "")),
                        prediction=norm(record_data.get("prediction", "")),
                        raw_output=record_data.get("raw_output"),
                        ground_truth=norm(record_data.get("ground_truth", "")),
                        score=float(record_data.get("score") or 0.0),
                        prompt_tokens=int(record_data.get("prompt_tokens") or 0),
                        completion_tokens=int(record_data.get("completion_tokens") or 0),
                        cost=float(record_data.get("cost") or 0.0),
                    )
                    total += 1
            except Exception as exc:
                logger.error(f"Failed to load {file_path}: {exc}")
        logger.info(f"Loaded {total} records")

    def load_all_records(self) -> List[BenchRecord]:
        return list(self.iter_records())

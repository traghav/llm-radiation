"""Weights & Biases integration for experiment tracking."""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

from llm_radiation.bitflip.manifest import FlipManifest
from llm_radiation.experiments.config import WandbConfig

logger = logging.getLogger(__name__)


class WandbTracker:
    """Wraps W&B for logging experiment metrics and artifacts."""

    def __init__(self, config: WandbConfig):
        self.config = config
        self._run = None

    def init(self, config: dict | None = None) -> None:
        """Initialize a W&B run."""
        import wandb

        self._run = wandb.init(
            project=self.config.project,
            entity=self.config.entity,
            config=config,
            tags=self.config.tags,
        )
        logger.info(f"W&B run initialized: {self._run.url}")

    def log_trial(self, record: dict) -> None:
        """Log a single trial's metrics."""
        if self._run is None:
            return

        flat = {
            "flip_rate": record["flip_rate"],
            "trial": record["trial"],
            "seed": record["seed"],
            "num_flips": record["num_flips"],
        }
        for bench_name, metrics in record["results"].items():
            for metric_name, value in metrics.items():
                flat[f"{bench_name}/{metric_name}"] = value

        self._run.log(flat)

    def log_manifest(self, manifest: FlipManifest, flip_rate: float, trial: int) -> None:
        """Log a flip manifest as a W&B artifact."""
        if self._run is None:
            return
        import wandb

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            manifest.to_json(f.name)
            artifact = wandb.Artifact(
                f"manifest-{flip_rate:.2e}-t{trial}",
                type="flip_manifest",
            )
            artifact.add_file(f.name, name="manifest.json")
            self._run.log_artifact(artifact)

    def finish(self) -> None:
        """Finish the W&B run."""
        if self._run is not None:
            self._run.finish()
            self._run = None

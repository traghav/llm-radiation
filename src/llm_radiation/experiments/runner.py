"""Main experiment loop: load model, sweep flip rates, benchmark, record."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import torch

from llm_radiation.bitflip.engine import BitFlipEngine
from llm_radiation.bitflip.manifest import FlipManifest
from llm_radiation.benchmarks.harness import BenchmarkRunner
from llm_radiation.experiments.config import ExperimentConfig
from llm_radiation.models.registry import load_model, snapshot_weights, restore_weights
from llm_radiation.tracking.wandb_tracker import WandbTracker

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Runs the full bit-flip degradation experiment for a single model."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.tracker: WandbTracker | None = None
        self.results: list[dict] = []

    def run(self) -> list[dict]:
        """Execute the full experiment sweep."""
        # Initialize tracking
        if self.config.wandb.enabled:
            self.tracker = WandbTracker(self.config.wandb)
            self.tracker.init(config=self.config.model_dump())

        # Load model and tokenizer
        logger.info(f"Loading model: {self.config.model.model_id}")
        model, tokenizer = load_model(
            self.config.model.model_id,
            dtype=self.config.model.dtype,
            device_map=self.config.model.device_map,
            revision=self.config.model.revision,
            trust_remote_code=self.config.model.trust_remote_code,
        )

        # Snapshot clean weights
        logger.info("Snapshotting clean weights to CPU...")
        clean_snapshot = snapshot_weights(model)
        model_hash = FlipManifest.compute_model_hash(clean_snapshot)

        # Setup benchmark runner
        bench_runner = BenchmarkRunner(self.config.benchmarks)

        # Setup bit flip engine
        engine = BitFlipEngine(seed=self.config.bitflip.seed_base)

        # Run baseline (0 flips)
        logger.info("Running baseline evaluation (no bit flips)...")
        baseline_results = bench_runner.run_all(model, tokenizer)
        baseline_record = {
            "flip_rate": 0.0,
            "trial": 0,
            "seed": 0,
            "num_flips": 0,
            "results": baseline_results,
            "timestamp": time.time(),
        }
        self.results.append(baseline_record)
        if self.tracker:
            self.tracker.log_trial(baseline_record)

        # Determine flip rates (possibly adaptive)
        flip_rates = self.config.bitflip.rates
        if self.config.adaptive.enabled and not flip_rates:
            from llm_radiation.experiments.adaptive import AdaptiveSampler
            sampler = AdaptiveSampler(self.config.adaptive)
            flip_rates = sampler.initial_rates()

        # Sweep flip rates
        for rate in flip_rates:
            for trial in range(self.config.bitflip.trials_per_rate):
                seed = self.config.bitflip.seed_base + hash((rate, trial)) % (2**31)

                logger.info(f"Flip rate={rate:.2e}, trial={trial}, seed={seed}")

                # Restore clean weights
                restore_weights(model, clean_snapshot)

                # Apply bit flips
                manifest = engine.apply_flips(
                    model,
                    flip_rate=rate,
                    model_id=self.config.model.model_id,
                    model_hash=model_hash,
                    seed=seed,
                )

                # Benchmark
                trial_results = bench_runner.run_all(model, tokenizer)

                record = {
                    "flip_rate": rate,
                    "trial": trial,
                    "seed": seed,
                    "num_flips": manifest.num_flips,
                    "results": trial_results,
                    "timestamp": time.time(),
                }
                self.results.append(record)

                if self.tracker:
                    self.tracker.log_trial(record)
                    self.tracker.log_manifest(manifest, rate, trial)

                logger.info(f"  Results: {trial_results}")

        # Adaptive densification
        if self.config.adaptive.enabled:
            from llm_radiation.experiments.adaptive import AdaptiveSampler
            sampler = AdaptiveSampler(self.config.adaptive)
            new_rates = sampler.densify(self.results)
            if new_rates:
                logger.info(f"Adaptive: adding {len(new_rates)} densified points")
                for rate in new_rates:
                    for trial in range(self.config.bitflip.trials_per_rate):
                        seed = self.config.bitflip.seed_base + hash((rate, trial)) % (2**31)
                        restore_weights(model, clean_snapshot)
                        manifest = engine.apply_flips(
                            model, flip_rate=rate,
                            model_id=self.config.model.model_id,
                            model_hash=model_hash, seed=seed,
                        )
                        trial_results = bench_runner.run_all(model, tokenizer)
                        record = {
                            "flip_rate": rate, "trial": trial, "seed": seed,
                            "num_flips": manifest.num_flips,
                            "results": trial_results, "timestamp": time.time(),
                        }
                        self.results.append(record)
                        if self.tracker:
                            self.tracker.log_trial(record)

        # Save results
        output_dir = Path(self.config.output_dir) / self.config.name
        output_dir.mkdir(parents=True, exist_ok=True)
        import json
        with open(output_dir / "results.json", "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        if self.tracker:
            self.tracker.finish()

        logger.info(f"Experiment complete. {len(self.results)} records saved to {output_dir}")
        return self.results

"""Tests for config system."""

import tempfile
from pathlib import Path

import pytest
import yaml

from llm_radiation.experiments.config import (
    ExperimentConfig,
    ModelConfig,
    BitFlipConfig,
    BenchmarkConfig,
    WandbConfig,
    AdaptiveConfig,
)


class TestExperimentConfig:
    def _make_config(self) -> ExperimentConfig:
        return ExperimentConfig(
            name="test_experiment",
            model=ModelConfig(model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
            bitflip=BitFlipConfig(rates=[1e-6, 1e-5, 1e-4], trials_per_rate=2),
            benchmarks=[
                BenchmarkConfig(name="hellaswag", tasks=["hellaswag"], num_fewshot=10),
            ],
            wandb=WandbConfig(enabled=False),
            adaptive=AdaptiveConfig(enabled=False),
        )

    def test_round_trip_yaml(self):
        config = self._make_config()
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            config.to_yaml(f.name)
            loaded = ExperimentConfig.from_yaml(f.name)

        assert loaded.name == config.name
        assert loaded.model.model_id == config.model.model_id
        assert loaded.bitflip.rates == config.bitflip.rates
        assert len(loaded.benchmarks) == 1

    def test_defaults(self):
        config = ExperimentConfig(
            name="test",
            model=ModelConfig(model_id="test-model"),
            bitflip=BitFlipConfig(),
            benchmarks=[],
        )
        assert config.model.dtype == "float16"
        assert config.bitflip.trials_per_rate == 3
        assert config.wandb.project == "llm-radiation"
        assert config.adaptive.enabled is True

    def test_model_config_validation(self):
        mc = ModelConfig(model_id="some/model", dtype="bfloat16")
        assert mc.dtype == "bfloat16"

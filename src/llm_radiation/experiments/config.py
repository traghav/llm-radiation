"""Configuration models for LLM radiation experiments.

This module defines Pydantic v2 models for all experiment configurations including
model settings, bit-flip parameters, benchmarks, tracking, and infrastructure.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Configuration for model loading and initialization."""

    model_id: str = Field(..., description="HuggingFace model ID or path")
    model_name: str = Field(default="", description="Human-readable name for the model")
    revision: str = Field(default="main", description="Model revision/branch")
    dtype: str = Field(default="float16", description="Data type for model weights")
    device_map: str = Field(default="auto", description="Device mapping strategy")
    trust_remote_code: bool = Field(
        default=False, description="Allow execution of remote code"
    )
    quantization: str = Field(
        default="none", description="Quantization method: 'none' or 'gptq'"
    )
    gptq_flip_targets: list[str] = Field(
        default_factory=lambda: ["qweight"],
        description="Which GPTQ tensors to target for bit flips",
    )


class BitFlipConfig(BaseModel):
    """Configuration for bit-flip injection experiments."""

    rates: list[float] = Field(default_factory=list, description="List of bit-flip rates to test")
    trials_per_rate: int = Field(default=3, description="Number of trials per rate")
    seed_base: int = Field(default=42, description="Base seed for reproducibility")


class BenchmarkConfig(BaseModel):
    """Configuration for a single benchmark evaluation."""

    name: str = Field(..., description="Benchmark name (e.g., 'mmlu', 'hellaswag')")
    tasks: list[str] = Field(..., description="List of tasks within the benchmark")
    num_fewshot: Optional[int] = Field(
        default=None, description="Number of few-shot examples"
    )
    limit: Optional[int] = Field(
        default=None, description="Limit number of samples (for quick testing)"
    )
    batch_size: int | str = Field(
        default="auto", description="Batch size for evaluation"
    )


class WandbConfig(BaseModel):
    """Configuration for Weights & Biases tracking."""

    project: str = Field(default="llm-radiation", description="W&B project name")
    entity: Optional[str] = Field(default=None, description="W&B entity/team name")
    enabled: bool = Field(default=True, description="Enable W&B logging")
    tags: list[str] = Field(default_factory=list, description="Tags for the run")


class LambdaConfig(BaseModel):
    """Configuration for Lambda Labs cloud infrastructure."""

    api_key: str = Field(default="", description="Lambda Labs API key")
    instance_type: str = Field(
        default="gpu_1x_a100", description="Instance type to provision"
    )
    region: str = Field(default="us-east-1", description="Region for deployment")
    ssh_key_name: str = Field(default="", description="SSH key name for access")


class AdaptiveConfig(BaseModel):
    """Configuration for adaptive bit-flip rate sampling."""

    enabled: bool = Field(default=True, description="Enable adaptive sampling")
    initial_points: int = Field(
        default=25, description="Number of initial sampling points"
    )
    min_rate: float = Field(default=1e-8, description="Minimum bit-flip rate")
    max_rate: float = Field(default=1e-2, description="Maximum bit-flip rate")
    gradient_threshold_multiplier: float = Field(
        default=2.0, description="Multiplier for gradient threshold detection"
    )
    max_recursion: int = Field(
        default=4, description="Maximum recursion depth for adaptive refinement"
    )

    def generate_initial_rates(self) -> list[float]:
        """Generate log-spaced initial sampling rates."""
        return np.logspace(
            np.log10(self.min_rate),
            np.log10(self.max_rate),
            self.initial_points,
        ).tolist()


class ExperimentConfig(BaseModel):
    """Complete configuration for a radiation experiment."""

    name: str = Field(..., description="Experiment name")
    model: ModelConfig = Field(..., description="Model configuration")
    bitflip: BitFlipConfig = Field(..., description="Bit-flip configuration")
    benchmarks: list[BenchmarkConfig] = Field(
        ..., description="List of benchmarks to run"
    )
    wandb: WandbConfig = Field(
        default_factory=WandbConfig, description="W&B tracking configuration"
    )
    lambda_labs: Optional[LambdaConfig] = Field(
        default=None, description="Lambda Labs configuration (optional)"
    )
    adaptive: AdaptiveConfig = Field(
        default_factory=AdaptiveConfig, description="Adaptive sampling configuration"
    )
    output_dir: str = Field(default="results", description="Output directory for results")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        """Load configuration from a YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            ExperimentConfig instance

        Raises:
            FileNotFoundError: If the config file doesn't exist
            ValueError: If the YAML is invalid or doesn't match the schema
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file.

        Args:
            path: Path where to save the YAML configuration
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict and handle Pydantic v2 serialization
        data = self.model_dump(mode="python", exclude_none=True)

        with open(path, "w") as f:
            yaml.safe_dump(
                data,
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )

    def get_all_rates(self) -> list[float]:
        """Get all bit-flip rates for the experiment.

        If adaptive is enabled and no rates are specified, generates initial rates.
        Otherwise returns the configured rates.
        """
        if not self.bitflip.rates and self.adaptive.enabled:
            return self.adaptive.generate_initial_rates()
        return self.bitflip.rates

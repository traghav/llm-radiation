"""FlipManifest for tracking and reproducing exact bit flip operations."""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import torch


@dataclass
class FlipRecord:
    """Record of a single bit flip."""

    param_name: str
    element_index: int
    bit_position: int


@dataclass
class FlipManifest:
    """Complete record of all bit flips applied to a model, for reproducibility."""

    model_id: str
    model_hash: str
    seed: int
    flip_rate: float
    total_bits: int
    num_flips: int
    flips: list[FlipRecord] = field(default_factory=list)

    def add_flip(self, param_name: str, element_index: int, bit_position: int) -> None:
        self.flips.append(FlipRecord(param_name, element_index, bit_position))

    def add_flips_bulk(
        self,
        param_name: str,
        element_indices: np.ndarray,
        bit_positions: np.ndarray,
    ) -> None:
        for ei, bp in zip(element_indices, bit_positions):
            self.flips.append(FlipRecord(param_name, int(ei), int(bp)))

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str | Path) -> FlipManifest:
        with open(path) as f:
            data = json.load(f)
        flips = [FlipRecord(**r) for r in data.pop("flips")]
        return cls(**data, flips=flips)

    @staticmethod
    def compute_model_hash(state_dict: dict) -> str:
        """Compute a deterministic hash of model weights for integrity verification."""
        h = hashlib.sha256()
        for name in sorted(state_dict.keys()):
            t = state_dict[name]
            h.update(name.encode())
            t_cpu = t.cpu()
            # Convert bfloat16 to float32 for numpy compat
            if t_cpu.dtype == torch.bfloat16:
                t_cpu = t_cpu.float()
            h.update(t_cpu.numpy().tobytes())
        return h.hexdigest()[:16]

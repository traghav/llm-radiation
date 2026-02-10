"""Unit tests for BitFlipEngine."""

import torch
import pytest

from llm_radiation.bitflip.engine import BitFlipEngine


class TestBitFlipEngine:
    def _make_small_model(self):
        """Create a tiny model for testing."""
        model = torch.nn.Sequential(
            torch.nn.Linear(16, 32),
            torch.nn.Linear(32, 8),
        ).half()  # Convert to fp16
        return model

    def test_count_total_bits(self):
        model = self._make_small_model()
        engine = BitFlipEngine(seed=42)
        total = engine.count_total_bits(model)
        # Linear(16,32): 16*32 + 32 = 544 params
        # Linear(32,8): 32*8 + 8 = 264 params
        # Total: 808 params * 16 bits = 12928 bits
        expected_params = 16 * 32 + 32 + 32 * 8 + 8
        assert total == expected_params * 16

    def test_apply_flips_changes_weights(self):
        model = self._make_small_model()
        engine = BitFlipEngine(seed=42)
        original_params = {n: p.data.clone() for n, p in model.named_parameters()}

        manifest = engine.apply_flips(model, flip_rate=0.01, model_id="test")

        # At least some parameters should have changed
        any_changed = False
        for name, param in model.named_parameters():
            if not torch.equal(param.data, original_params[name]):
                any_changed = True
                break
        assert any_changed

    def test_manifest_records_correct_count(self):
        model = self._make_small_model()
        engine = BitFlipEngine(seed=42)
        manifest = engine.apply_flips(model, flip_rate=0.001, model_id="test")

        expected_flips = max(1, int(engine.count_total_bits(model) * 0.001))
        assert manifest.num_flips == expected_flips
        assert len(manifest.flips) == expected_flips

    def test_reproducibility(self):
        model1 = self._make_small_model()
        model2 = self._make_small_model()
        # Ensure same initial weights
        model2.load_state_dict(model1.state_dict())

        engine = BitFlipEngine(seed=42)
        engine.apply_flips(model1, flip_rate=0.01, model_id="test", seed=123)
        engine.apply_flips(model2, flip_rate=0.01, model_id="test", seed=123)

        for (_, p1), (_, p2) in zip(model1.named_parameters(), model2.named_parameters()):
            assert torch.equal(p1.data, p2.data)

    def test_different_seeds_different_results(self):
        model1 = self._make_small_model()
        model2 = self._make_small_model()
        model2.load_state_dict(model1.state_dict())

        engine = BitFlipEngine()
        engine.apply_flips(model1, flip_rate=0.01, model_id="test", seed=1)
        engine.apply_flips(model2, flip_rate=0.01, model_id="test", seed=2)

        any_different = False
        for (_, p1), (_, p2) in zip(model1.named_parameters(), model2.named_parameters()):
            if not torch.equal(p1.data, p2.data):
                any_different = True
                break
        assert any_different

    def test_zero_rate_minimal_flips(self):
        """Even at very low rate, at least 1 flip should happen."""
        model = self._make_small_model()
        engine = BitFlipEngine(seed=42)
        manifest = engine.apply_flips(model, flip_rate=1e-10, model_id="test")
        assert manifest.num_flips >= 1

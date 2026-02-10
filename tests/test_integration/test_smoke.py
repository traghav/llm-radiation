"""Integration smoke test - requires GPU and model download."""

import pytest
import torch

# Skip all tests in this module if no GPU available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Integration tests require GPU",
)


class TestSmoke:
    """End-to-end smoke test with a tiny model."""

    @pytest.mark.slow
    def test_full_pipeline_tiny_model(self):
        """Full pipeline: load model -> flip bits -> evaluate.

        This test downloads TinyLlama and runs a minimal benchmark.
        Skip in CI with: pytest -m "not slow"
        """
        from llm_radiation.models.registry import load_model, snapshot_weights, restore_weights
        from llm_radiation.bitflip.engine import BitFlipEngine

        model, tokenizer = load_model("tinyllama")
        snapshot = snapshot_weights(model)

        engine = BitFlipEngine(seed=42)
        manifest = engine.apply_flips(model, flip_rate=1e-6, model_id="tinyllama")

        assert manifest.num_flips > 0
        assert len(manifest.flips) == manifest.num_flips

        # Verify weights actually changed
        any_changed = False
        for name, param in model.named_parameters():
            if name in snapshot and not torch.equal(param.data.cpu(), snapshot[name]):
                any_changed = True
                break
        assert any_changed

        # Restore and verify
        restore_weights(model, snapshot)
        for name, param in model.named_parameters():
            if name in snapshot:
                assert torch.equal(param.data.cpu(), snapshot[name])

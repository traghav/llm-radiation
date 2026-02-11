"""Unit tests for BitFlipEngine GPTQ support."""

import torch
import pytest

from llm_radiation.bitflip.engine import BitFlipEngine


class MockQuantLinear(torch.nn.Module):
    """Mock GPTQ QuantLinear layer with qweight, qzeros, scales, g_idx."""

    def __init__(self, in_features: int, out_features: int, group_size: int = 128):
        super().__init__()
        # qweight: int32 packed weights. For 4-bit with group packing:
        # shape is (in_features // 8, out_features) for 4-bit packed into int32
        pack_factor = 8  # 32 bits / 4 bits per weight
        qweight_rows = in_features // pack_factor
        self.register_buffer(
            "qweight", torch.randint(-2**31, 2**31 - 1, (qweight_rows, out_features), dtype=torch.int32)
        )
        n_groups = in_features // group_size
        self.register_buffer(
            "qzeros", torch.randint(0, 16, (n_groups, out_features // pack_factor), dtype=torch.int32)
        )
        self.register_buffer(
            "scales", torch.randn(n_groups, out_features, dtype=torch.float16)
        )
        self.register_buffer(
            "g_idx", torch.arange(in_features, dtype=torch.int32)
        )


class MockGPTQModel(torch.nn.Module):
    """Mock model with QuantLinear layers."""

    def __init__(self):
        super().__init__()
        self.layer1 = MockQuantLinear(256, 128)
        self.layer2 = MockQuantLinear(128, 64)
        # Also have a normal float param (like lm_head or embeddings)
        self.lm_head = torch.nn.Linear(64, 100).half()


class TestBitFlipEngineGPTQ:
    def _make_model(self):
        return MockGPTQModel()

    def test_count_gptq_bits(self):
        model = self._make_model()
        engine = BitFlipEngine(seed=42)
        total = engine.count_total_bits(model, quantization="gptq")
        # layer1.qweight: (256//8, 128) = (32, 128) = 4096 elements * 32 bits
        # layer2.qweight: (128//8, 64) = (16, 64) = 1024 elements * 32 bits
        expected = (4096 + 1024) * 32
        assert total == expected

    def test_count_gptq_bits_multiple_targets(self):
        model = self._make_model()
        engine = BitFlipEngine(seed=42)
        total = engine.count_total_bits(
            model, quantization="gptq", gptq_flip_targets=["qweight", "qzeros"]
        )
        # qweight as above: (4096 + 1024) * 32
        # qzeros: layer1 (2, 16) = 32 elements, layer2 (1, 8) = 8 elements
        qweight_bits = (4096 + 1024) * 32
        qzeros_bits = (32 + 8) * 32
        assert total == qweight_bits + qzeros_bits

    def test_apply_gptq_flips_changes_qweight(self):
        model = self._make_model()
        engine = BitFlipEngine(seed=42)

        # Snapshot original qweights
        orig_qw1 = model.layer1.qweight.clone()
        orig_qw2 = model.layer2.qweight.clone()

        manifest = engine.apply_flips(
            model, flip_rate=0.01, model_id="test",
            quantization="gptq",
        )

        # At least one qweight should have changed
        changed = (
            not torch.equal(model.layer1.qweight, orig_qw1)
            or not torch.equal(model.layer2.qweight, orig_qw2)
        )
        assert changed
        assert manifest.num_flips > 0

    def test_gptq_flips_dont_touch_fp16(self):
        """GPTQ flip mode should not modify FP16 parameters."""
        model = self._make_model()
        engine = BitFlipEngine(seed=42)

        orig_lm_head = {n: p.data.clone() for n, p in model.lm_head.named_parameters()}

        engine.apply_flips(
            model, flip_rate=0.01, model_id="test",
            quantization="gptq",
        )

        for name, param in model.lm_head.named_parameters():
            assert torch.equal(param.data, orig_lm_head[name])

    def test_gptq_reproducibility(self):
        model1 = self._make_model()
        model2 = self._make_model()
        # Ensure same initial state
        model2.load_state_dict(model1.state_dict())

        engine = BitFlipEngine(seed=42)
        engine.apply_flips(model1, flip_rate=0.01, model_id="test", seed=123, quantization="gptq")
        engine.apply_flips(model2, flip_rate=0.01, model_id="test", seed=123, quantization="gptq")

        assert torch.equal(model1.layer1.qweight, model2.layer1.qweight)
        assert torch.equal(model1.layer2.qweight, model2.layer2.qweight)

    def test_manifest_records_correct_count(self):
        model = self._make_model()
        engine = BitFlipEngine(seed=42)
        manifest = engine.apply_flips(
            model, flip_rate=0.001, model_id="test", quantization="gptq"
        )
        expected_total = engine.count_total_bits(model, quantization="gptq")
        expected_flips = max(1, int(expected_total * 0.001))
        assert manifest.num_flips == expected_flips
        assert len(manifest.flips) == expected_flips

    def test_health_check_gptq_skips_int32(self):
        """Health check in GPTQ mode should not crash on int32 qweight tensors."""
        model = self._make_model()
        report = BitFlipEngine.check_health(model, quantization="gptq")
        # Should report on FP16 params (lm_head) but not crash on int32
        assert report.total_params > 0
        assert report.weights_healthy

    def test_fp16_count_unchanged(self):
        """FP16 counting should still work and ignore int32 tensors."""
        model = self._make_model()
        engine = BitFlipEngine(seed=42)
        fp16_bits = engine.count_total_bits(model, quantization="none")
        # Only lm_head params: Linear(64, 100) = 64*100 + 100 = 6500 params * 16 bits
        expected = (64 * 100 + 100) * 16
        assert fp16_bits == expected

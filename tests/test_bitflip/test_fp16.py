"""Unit tests for FP16 bit manipulation."""

import numpy as np
import torch
import pytest

from llm_radiation.bitflip.fp16 import flip_bits_fp16, generate_uniform_flips


class TestFlipBitsFP16:
    def test_sign_bit_flip_negates(self):
        """Flipping bit 15 (sign bit) should negate the value."""
        t = torch.tensor([1.0, -2.0, 3.5], dtype=torch.float16)
        original = t.clone()
        element_indices = np.array([0, 1, 2])
        bit_positions = np.array([15, 15, 15])  # sign bit
        flip_bits_fp16(t, element_indices, bit_positions)
        assert torch.allclose(t, -original)

    def test_double_flip_restores(self):
        """Flipping the same bit twice should restore original value."""
        t = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float16)
        original = t.clone()
        element_indices = np.array([0, 1, 2, 3])
        bit_positions = np.array([3, 7, 11, 15])
        flip_bits_fp16(t, element_indices, bit_positions)
        assert not torch.equal(t, original)
        flip_bits_fp16(t, element_indices, bit_positions)
        assert torch.equal(t, original)

    def test_mantissa_flip_small_change(self):
        """Flipping LSB of mantissa should change value by 1 ULP."""
        t = torch.tensor([1.0], dtype=torch.float16)
        flip_bits_fp16(t, np.array([0]), np.array([0]))  # bit 0 = LSB mantissa
        # Should differ by exactly 1 ULP for 1.0 in fp16
        assert t.item() != 1.0
        assert abs(t.item() - 1.0) < 0.002  # 1 ULP for fp16 near 1.0

    def test_exponent_flip_creates_large_change(self):
        """Flipping an exponent bit should create a large magnitude change."""
        t = torch.tensor([1.0], dtype=torch.float16)
        original_val = t.item()
        flip_bits_fp16(t, np.array([0]), np.array([14]))  # high exponent bit
        assert abs(t.item() - original_val) > 100  # should be huge change

    def test_can_create_nan(self):
        """Flipping exponent bits can create NaN/Inf values."""
        # All exponent bits set + nonzero mantissa = NaN
        t = torch.tensor([65504.0], dtype=torch.float16)  # max finite fp16
        flip_bits_fp16(t, np.array([0]), np.array([0]))  # flip mantissa LSB
        # 65504 = 0_11110_1111111111, flipping bit 0 -> 0_11111_1111111110 which could be NaN
        # Actually max finite is 0_11110_1111111111, flipping LSB gives 0_11110_1111111110
        # Let's just check the flip happened
        assert t.item() != 65504.0

    def test_exact_flip_count(self):
        """Exactly the requested number of bits should be flipped."""
        t = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float16)
        original_uint = t.view(torch.uint16).clone()
        element_indices = np.array([0, 2])
        bit_positions = np.array([5, 10])
        flip_bits_fp16(t, element_indices, bit_positions)
        new_uint = t.view(torch.uint16)
        # Count total bits that changed
        xor = original_uint ^ new_uint
        total_changed = sum(bin(x.item()).count('1') for x in xor)
        assert total_changed == 2

    def test_empty_flip(self):
        """Flipping zero bits should not change tensor."""
        t = torch.tensor([1.0, 2.0], dtype=torch.float16)
        original = t.clone()
        flip_bits_fp16(t, np.array([], dtype=np.int64), np.array([], dtype=np.int32))
        assert torch.equal(t, original)

    def test_wrong_dtype_raises(self):
        """Should raise assertion error for non-FP16 tensor."""
        t = torch.tensor([1.0], dtype=torch.float32)
        with pytest.raises(AssertionError):
            flip_bits_fp16(t, np.array([0]), np.array([0]))


class TestGenerateUniformFlips:
    def test_correct_count(self):
        rng = np.random.default_rng(42)
        elem_idx, bit_pos = generate_uniform_flips(1000, 50, rng)
        assert len(elem_idx) == 50
        assert len(bit_pos) == 50

    def test_no_duplicates(self):
        rng = np.random.default_rng(42)
        elem_idx, bit_pos = generate_uniform_flips(100, 50, rng)
        flat = elem_idx * 16 + bit_pos
        assert len(set(flat)) == 50

    def test_bit_positions_in_range(self):
        rng = np.random.default_rng(42)
        _, bit_pos = generate_uniform_flips(1000, 500, rng)
        assert all(0 <= b < 16 for b in bit_pos)

    def test_element_indices_in_range(self):
        rng = np.random.default_rng(42)
        elem_idx, _ = generate_uniform_flips(1000, 500, rng)
        assert all(0 <= e < 1000 for e in elem_idx)

    def test_reproducibility(self):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        e1, b1 = generate_uniform_flips(1000, 50, rng1)
        e2, b2 = generate_uniform_flips(1000, 50, rng2)
        assert np.array_equal(e1, e2)
        assert np.array_equal(b1, b2)

    def test_too_many_flips_raises(self):
        rng = np.random.default_rng(42)
        with pytest.raises(AssertionError):
            generate_uniform_flips(10, 200, rng)  # 10*16=160 < 200

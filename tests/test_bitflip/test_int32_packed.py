"""Unit tests for int32 packed bit manipulation (GPTQ qweight)."""

import numpy as np
import torch
import pytest

from llm_radiation.bitflip.int32_packed import flip_bits_int32


class TestFlipBitsInt32:
    def test_double_flip_restores(self):
        """Flipping the same bit twice should restore original value."""
        t = torch.tensor([12345, -67890, 0, 2**30], dtype=torch.int32)
        original = t.clone()
        element_indices = np.array([0, 1, 2, 3])
        bit_positions = np.array([3, 15, 0, 31])
        flip_bits_int32(t, element_indices, bit_positions)
        assert not torch.equal(t, original)
        flip_bits_int32(t, element_indices, bit_positions)
        assert torch.equal(t, original)

    def test_exact_flip_count(self):
        """Exactly the requested number of bits should be flipped."""
        t = torch.tensor([0, 0, 0, 0], dtype=torch.int32)
        original = t.clone()
        element_indices = np.array([0, 2])
        bit_positions = np.array([5, 20])
        flip_bits_int32(t, element_indices, bit_positions)
        # Element 0 should have bit 5 set, element 2 should have bit 20 set
        assert t[0].item() == (1 << 5)
        assert t[1].item() == 0  # untouched
        assert t[2].item() == (1 << 20)
        assert t[3].item() == 0  # untouched

    def test_specific_bit_flip(self):
        """Flipping bit 0 on value 0 should give 1."""
        t = torch.tensor([0], dtype=torch.int32)
        flip_bits_int32(t, np.array([0]), np.array([0]))
        assert t[0].item() == 1

    def test_high_bit_flip(self):
        """Flipping bit 31 (sign bit for signed int32) should change sign."""
        t = torch.tensor([0], dtype=torch.int32)
        flip_bits_int32(t, np.array([0]), np.array([31]))
        # Setting bit 31 on 0 gives -2^31 in two's complement
        assert t[0].item() == -(2**31)

    def test_empty_flip(self):
        """Flipping zero bits should not change tensor."""
        t = torch.tensor([42, 99], dtype=torch.int32)
        original = t.clone()
        flip_bits_int32(t, np.array([], dtype=np.int64), np.array([], dtype=np.int32))
        assert torch.equal(t, original)

    def test_wrong_dtype_raises(self):
        """Should raise assertion error for non-int32 tensor."""
        t = torch.tensor([1.0], dtype=torch.float32)
        with pytest.raises(AssertionError):
            flip_bits_int32(t, np.array([0]), np.array([0]))

    def test_multiple_elements_different_bits(self):
        """Flips on different elements with different bits should all apply."""
        t = torch.tensor([0, 0, 0], dtype=torch.int32)
        element_indices = np.array([0, 1, 2])
        bit_positions = np.array([0, 1, 2])
        flip_bits_int32(t, element_indices, bit_positions)
        assert t[0].item() == (1 << 0)
        assert t[1].item() == (1 << 1)
        assert t[2].item() == (1 << 2)

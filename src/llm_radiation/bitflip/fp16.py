"""Raw FP16 bit manipulation via uint16 view + XOR."""

import torch
import numpy as np
from numpy.random import Generator


def flip_bits_fp16(
    tensor: torch.Tensor,
    element_indices: np.ndarray,
    bit_positions: np.ndarray,
) -> None:
    """Flip specific bits in an FP16 tensor via zero-copy uint16 view.

    Uses numpy for the actual XOR since PyTorch's index_put doesn't support
    UInt16 on CPU. The numpy array shares memory with the torch tensor.

    Args:
        tensor: A contiguous float16 parameter tensor (modified in-place).
        element_indices: Flat indices into the tensor specifying which elements to flip.
        bit_positions: Which bit (0-15) to flip in each element. Must be same length
            as element_indices.
    """
    assert tensor.dtype == torch.float16, f"Expected float16 tensor, got {tensor.dtype}"
    assert tensor.is_contiguous(), "Tensor must be contiguous for uint16 view"
    assert len(element_indices) == len(bit_positions)

    if len(element_indices) == 0:
        return

    # Move to CPU for numpy interop if needed
    was_cuda = tensor.is_cuda
    if was_cuda:
        # For CUDA tensors, use torch operations
        flat = tensor.view(-1).view(torch.int16)  # int16 is supported unlike uint16
        idx = torch.from_numpy(element_indices.astype(np.int64)).to(flat.device)
        masks = torch.from_numpy((1 << bit_positions).astype(np.int16)).to(flat.device)
        flat[idx] ^= masks
    else:
        # For CPU tensors, use numpy zero-copy view for best compatibility
        np_view = tensor.view(-1).numpy().view(np.uint16)
        masks = (1 << bit_positions).astype(np.uint16)
        np_view[element_indices] ^= masks


def generate_uniform_flips(
    num_elements: int,
    num_flips: int,
    rng: Generator,
    bits_per_element: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate uniformly random flip locations.

    Args:
        num_elements: Total number of FP16 elements in the tensor.
        num_flips: Number of bits to flip.
        rng: Numpy random Generator for reproducibility.
        bits_per_element: Bits per element (16 for FP16).

    Returns:
        Tuple of (element_indices, bit_positions) arrays.
    """
    total_bits = num_elements * bits_per_element
    assert num_flips <= total_bits, (
        f"Cannot flip {num_flips} bits in {total_bits} total bits"
    )

    flat_bit_indices = rng.choice(total_bits, size=num_flips, replace=False)
    element_indices = (flat_bit_indices // bits_per_element).astype(np.int64)
    bit_positions = (flat_bit_indices % bits_per_element).astype(np.int32)

    return element_indices, bit_positions

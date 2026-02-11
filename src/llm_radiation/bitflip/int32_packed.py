"""Raw int32 bit manipulation for GPTQ packed int4 weights via XOR."""

import torch
import numpy as np


def flip_bits_int32(
    tensor: torch.Tensor,
    element_indices: np.ndarray,
    bit_positions: np.ndarray,
) -> None:
    """Flip specific bits in an int32 tensor (GPTQ qweight) via XOR.

    Args:
        tensor: A contiguous int32 tensor (modified in-place).
        element_indices: Flat indices into the tensor specifying which elements to flip.
        bit_positions: Which bit (0-31) to flip in each element. Must be same length
            as element_indices.
    """
    assert tensor.dtype == torch.int32, f"Expected int32 tensor, got {tensor.dtype}"
    assert tensor.is_contiguous(), "Tensor must be contiguous for int32 view"
    assert len(element_indices) == len(bit_positions)

    if len(element_indices) == 0:
        return

    if tensor.is_cuda:
        flat = tensor.view(-1)
        idx = torch.from_numpy(element_indices.astype(np.int64)).to(flat.device)
        masks = torch.from_numpy((1 << bit_positions).astype(np.int32)).to(flat.device)
        flat[idx] ^= masks
    else:
        np_view = tensor.view(-1).numpy().view(np.int32)
        masks = (1 << bit_positions).astype(np.int32)
        np_view[element_indices] ^= masks

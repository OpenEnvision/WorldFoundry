import torch
import torch.nn.functional as F


def _pad_end(tensor: torch.Tensor, dim: int, length: int, value: float) -> torch.Tensor:
    """Constant-pad the end of one dimension using reversed (start, end) pairs."""
    padding_dims = [0] * (2 * tensor.dim())
    pad_index = 2 * (tensor.dim() - dim - 1) + 1
    if pad_index < len(padding_dims):
        padding_dims[pad_index] = length
    else:
        raise ValueError("Invalid dimension index.")
    return F.pad(tensor, pad=tuple(padding_dims), mode="constant", value=value)


def minimal_pad_to_divisible(tensor: torch.Tensor, sp_size: int, dim: int = 1, pad_value: float = 0.0):
    """Pad the end of dim with the fewest elements needed for divisibility by sp_size.

    Return (padded_tensor, padding_len), retaining the input object when no
    padding is needed. dim defaults to the second dimension and pad_value to
    zero. Negative dimensions are accepted only when no padding is needed."""

    current_size = tensor.size(dim)
    padding_len = (sp_size - current_size % sp_size) % sp_size

    if padding_len == 0:
        return tensor, 0
    return _pad_end(tensor, dim, padding_len, pad_value), padding_len


def depad_by_length(padded_tensor: torch.Tensor, depadding_len: int, dim: int = 1) -> torch.Tensor:
    """Return a view without the last depadding_len elements of dim (the second dimension by default).

    Raise ValueError for a negative removal length or one exceeding the dimension size."""
    current_size = padded_tensor.size(dim)
    if depadding_len < 0:
        raise ValueError("depadding_len must be non-negative.")
    if depadding_len > current_size:
        raise ValueError(f"Cannot remove {depadding_len} elements from a dimension of size {current_size}.")
    target_size = current_size - depadding_len
    slices = [slice(None)] * padded_tensor.dim()
    slices[dim] = slice(0, target_size)
    depadded_tensor = padded_tensor[tuple(slices)]

    return depadded_tensor


def pad_by_length(padded_tensor: torch.Tensor, padding_len: int, dim: int = 1, pad_value: float = 0.0) -> torch.Tensor:
    if padding_len < 0:
        raise ValueError("padding_len must be non-negative.")

    if dim < 0 or dim >= padded_tensor.dim():
        raise ValueError(f"Dimension index {dim} is outside [0, {padded_tensor.dim() - 1}].")
    return _pad_end(padded_tensor, dim, padding_len, pad_value)

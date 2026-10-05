"""Convolution and pooling shape inference for numpy and torch tensors.

Responsibility
    Predict output spatial size for conv / pool / transpose-conv so graph
    builders and tilers can allocate without running the op.

Boundaries
    Not a replacement for ``F.conv2d``: it returns a shape tuple only, never
    a tensor. Formulas match PyTorch ``nn.Conv*`` / ``nn.ConvTranspose*`` /
    pooling docs. ``has_batch`` decides whether the leading dim is batch.
    ``check_shape`` accepts ``None`` in the expected tuple as a wildcard dim.

Public surface
    ``shape_conv{1,2,3}d``, ``shape_transpose_conv{1,2,3}d``,
    ``shape_{max,avg}pool{1,2,3}d``, :func:`shape_slice`, :func:`check_shape`,
    :class:`ShapeSlice`.

Formulas::

    conv / pool:  out = floor((in + 2*pad - dilation*(kernel-1) - 1) / stride) + 1
    transpose:    out = (in - 1)*stride - 2*pad + kernel + output_padding
"""

import math
import warnings
from functools import partial
from typing import List, Tuple, Union

import numpy as np
import torch

# fmt: off
__all__ = [
    "shape_convnd",
    "shape_conv1d", "shape_conv2d", "shape_conv3d",
    "shape_transpose_convnd",
    "shape_transpose_conv1d", "shape_transpose_conv2d", "shape_transpose_conv3d",
    "shape_poolnd",
    "shape_maxpool1d", "shape_maxpool2d", "shape_maxpool3d",
    "shape_avgpool1d", "shape_avgpool2d", "shape_avgpool3d",
    "shape_slice",
    "check_shape"
]
# fmt: on


# ──────────────────────────────────────────────────────────────────────────
# Shape accessors — numpy / torch leaves only; no pytree walk
# ──────────────────────────────────────────────────────────────────────────


def _get_shape(x):
    """Return ``ndarray.shape`` or ``Tensor.size()`` as a tuple."""
    if isinstance(x, np.ndarray):
        return tuple(x.shape)
    else:
        return tuple(x.size())


def _expands(dim, *xs):
    """Broadcast an int kernel/stride/padding to a ``dim``-tuple, or assert length."""

    def _expand(x):
        """Broadcast a scalar kernel/stride/pad to ``dim`` axes, or assert length."""
        if isinstance(x, int):
            return (x,) * dim
        else:
            assert len(x) == dim
            return x

    return map(lambda x: _expand(x), xs)


_HELPER_TENSOR = torch.zeros((1,))


def shape_slice(input_shape, slice):
    """
    Credit to Adam Paszke for the trick. Shape inference without instantiating
    an actual tensor.
    The key is that `.expand()` does not actually allocate memory
    Still needs to allocate a one-element HELPER_TENSOR.
    """
    shape = _HELPER_TENSOR.expand(*input_shape)[slice]
    if hasattr(shape, "size"):
        return tuple(shape.size())
    return (1,)


class ShapeSlice:
    """
    shape_slice inference with easy []-operator
    """

    def __init__(self, input_shape):
        """Hold ``input_shape`` so ``shape_slice[idx]`` can run without a tensor."""
        self.input_shape = input_shape

    def __getitem__(self, slice):
        """Infer the result shape of ``tensor[slice]`` via a 1-element expand."""
        return shape_slice(self.input_shape, slice)


def check_shape(
    value: Union[Tuple, List, torch.Tensor, np.ndarray],
    expected: Union[Tuple, List, torch.Tensor, np.ndarray],
    err_msg="",
    mode="raise",
):
    """
    Args:
        value: np array or torch Tensor
        expected:
          - list[int], tuple[int]: if any value is None, will match any dim
          - np array or torch Tensor: must have the same dimensions
        mode:
          - "raise": raise ValueError, shape mismatch
          - "return": returns True if shape matches, otherwise False
          - "warning": warnings.warn
    """
    assert mode in ["raise", "return", "warning"]
    if torch.is_tensor(value):
        actual_shape = value.size()
    elif hasattr(value, "shape"):
        actual_shape = value.shape
    else:
        assert isinstance(value, (list, tuple))
        actual_shape = value
        assert all(isinstance(s, int) for s in actual_shape), f"actual shape: {actual_shape} is not a list of ints"

    if torch.is_tensor(expected):
        expected_shape = expected.size()
    elif hasattr(expected, "shape"):
        expected_shape = expected.shape
    else:
        assert isinstance(expected, (list, tuple))
        expected_shape = expected

    err_msg = f" for {err_msg}" if err_msg else ""

    if len(actual_shape) != len(expected_shape):
        err_msg = f"Dimension mismatch{err_msg}: actual shape {actual_shape} != expected shape {expected_shape}."
        if mode == "raise":
            raise ValueError(err_msg)
        elif mode == "warning":
            warnings.warn(err_msg)
        return False

    for s_a, s_e in zip(actual_shape, expected_shape):
        if s_e is not None and s_a != s_e:
            err_msg = f"Shape mismatch{err_msg}: actual shape {actual_shape} != expected shape {expected_shape}."
            if mode == "raise":
                raise ValueError(err_msg)
            elif mode == "warning":
                warnings.warn(err_msg)
            return False
    return True


def shape_convnd(
    dim,
    input_shape,
    out_channels,
    kernel_size,
    stride=1,
    padding=0,
    dilation=1,
    has_batch=False,
):
    """
    http://pytorch.org/docs/nn.html#conv1d
    http://pytorch.org/docs/nn.html#conv2d
    http://pytorch.org/docs/nn.html#conv3d

    Args:
        dim: supports 1D to 3D
        input_shape:
        - 1D: [channel, length]
        - 2D: [channel, height, width]
        - 3D: [channel, depth, height, width]
        has_batch: whether the first dim is batch size or not
    """
    if has_batch:
        assert len(input_shape) == dim + 2, "input shape with batch should be {}-dimensional".format(dim + 2)
    else:
        assert len(input_shape) == dim + 1, "input shape without batch should be {}-dimensional".format(dim + 1)
    if stride is None:
        # for pooling convention in PyTorch
        stride = kernel_size
    kernel_size, stride, padding, dilation = _expands(dim, kernel_size, stride, padding, dilation)
    if has_batch:
        batch = input_shape[0]
        input_shape = input_shape[1:]
    else:
        batch = None
    _, *img = input_shape
    # PyTorch conv output: floor((L + 2P - D*(K-1) - 1) / S + 1) per spatial axis.
    new_img_shape = [
        math.floor((img[i] + 2 * padding[i] - dilation[i] * (kernel_size[i] - 1) - 1) // stride[i] + 1)
        for i in range(dim)
    ]
    return ((batch,) if has_batch else ()) + (out_channels, *new_img_shape)


def shape_poolnd(dim, input_shape, kernel_size, stride=None, padding=0, dilation=1, has_batch=False):
    """
    The only difference from infer_shape_convnd is that `stride` default is None
    """
    if has_batch:
        out_channels = input_shape[1]
    else:
        out_channels = input_shape[0]
    return shape_convnd(
        dim,
        input_shape,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        has_batch,
    )


def shape_transpose_convnd(
    dim,
    input_shape,
    out_channels,
    kernel_size,
    stride=1,
    padding=0,
    output_padding=0,
    dilation=1,
    has_batch=False,
):
    """
    http://pytorch.org/docs/nn.html#convtranspose1d
    http://pytorch.org/docs/nn.html#convtranspose2d
    http://pytorch.org/docs/nn.html#convtranspose3d

    Args:
        dim: supports 1D to 3D
        input_shape:
        - 1D: [channel, length]
        - 2D: [channel, height, width]
        - 3D: [channel, depth, height, width]
        has_batch: whether the first dim is batch size or not
    """
    if has_batch:
        assert len(input_shape) == dim + 2, "input shape with batch should be {}-dimensional".format(dim + 2)
    else:
        assert len(input_shape) == dim + 1, "input shape without batch should be {}-dimensional".format(dim + 1)
    kernel_size, stride, padding, output_padding, dilation = _expands(
        dim, kernel_size, stride, padding, output_padding, dilation
    )
    if has_batch:
        batch = input_shape[0]
        input_shape = input_shape[1:]
    else:
        batch = None
    _, *img = input_shape
    # Transpose-conv: (L-1)*S - 2P + K + output_padding (dilation absorbed into K).
    new_img_shape = [(img[i] - 1) * stride[i] - 2 * padding[i] + kernel_size[i] + output_padding[i] for i in range(dim)]
    return ((batch,) if has_batch else ()) + (out_channels, *new_img_shape)


shape_conv1d = partial(shape_convnd, 1)
shape_conv2d = partial(shape_convnd, 2)
shape_conv3d = partial(shape_convnd, 3)


shape_transpose_conv1d = partial(shape_transpose_convnd, 1)
shape_transpose_conv2d = partial(shape_transpose_convnd, 2)
shape_transpose_conv3d = partial(shape_transpose_convnd, 3)


shape_maxpool1d = partial(shape_poolnd, 1)
shape_maxpool2d = partial(shape_poolnd, 2)
shape_maxpool3d = partial(shape_poolnd, 3)


"""
http://pytorch.org/docs/nn.html#avgpool1d
http://pytorch.org/docs/nn.html#avgpool2d
http://pytorch.org/docs/nn.html#avgpool3d
"""
shape_avgpool1d = partial(shape_maxpool1d, dilation=1)
shape_avgpool2d = partial(shape_maxpool2d, dilation=1)
shape_avgpool3d = partial(shape_maxpool3d, dilation=1)

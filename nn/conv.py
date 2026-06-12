"""Convolutional building blocks: Conv2D, MaxPool2D, Flatten.

These layers operate on 4-D tensors shaped ``(N, C, H, W)`` (batch, channels,
height, width), the convention used throughout. ``Flatten`` bridges them to the
2-D ``(N, features)`` world that :class:`~nn.layers.Dense` expects.

``Conv2D`` uses the classic *im2col* trick: each receptive field is unrolled
into a row so the convolution becomes a single matrix multiply. This keeps the
whole thing vectorised NumPy with no Python inner loops over pixels.
"""

import numpy as np

from .layers import Layer


def _conv_out_size(size, kernel, stride, pad):
    return (size + 2 * pad - kernel) // stride + 1


def im2col(x, kh, kw, stride, pad):
    """Unroll receptive fields into columns.

    Returns an array of shape ``(N*out_h*out_w, C*kh*kw)``.
    """
    n, c, h, w = x.shape
    out_h = _conv_out_size(h, kh, stride, pad)
    out_w = _conv_out_size(w, kw, stride, pad)

    x_padded = np.pad(
        x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant"
    )
    cols = np.empty((n, c, kh, kw, out_h, out_w), dtype=x.dtype)
    for i in range(kh):
        i_max = i + stride * out_h
        for j in range(kw):
            j_max = j + stride * out_w
            cols[:, :, i, j, :, :] = x_padded[:, :, i:i_max:stride, j:j_max:stride]

    # (N, out_h, out_w, C, kh, kw) -> rows of length C*kh*kw
    cols = cols.transpose(0, 4, 5, 1, 2, 3).reshape(n * out_h * out_w, -1)
    return cols


def col2im(cols, x_shape, kh, kw, stride, pad):
    """Inverse of :func:`im2col`, scattering column gradients back to pixels."""
    n, c, h, w = x_shape
    out_h = _conv_out_size(h, kh, stride, pad)
    out_w = _conv_out_size(w, kw, stride, pad)

    cols = cols.reshape(n, out_h, out_w, c, kh, kw).transpose(0, 3, 4, 5, 1, 2)
    x_padded = np.zeros((n, c, h + 2 * pad, w + 2 * pad), dtype=cols.dtype)
    for i in range(kh):
        i_max = i + stride * out_h
        for j in range(kw):
            j_max = j + stride * out_w
            # Overlapping fields accumulate, hence += not =.
            x_padded[:, :, i:i_max:stride, j:j_max:stride] += cols[:, :, i, j, :, :]

    if pad == 0:
        return x_padded
    return x_padded[:, :, pad:-pad, pad:-pad]


class Conv2D(Layer):
    """2-D convolution over ``(N, C, H, W)`` inputs.

    Parameters
    ----------
    in_channels, out_channels : int
    kernel_size : int
        Square kernel side length.
    stride : int
    padding : int
        Zero-padding on every side. ``padding = kernel_size // 2`` with
        ``stride = 1`` keeps the spatial size unchanged ("same" convolution).
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, seed=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = kernel_size
        self.stride = stride
        self.pad = padding

        rng = np.random.default_rng(seed)
        fan_in = in_channels * kernel_size * kernel_size
        scale = np.sqrt(2.0 / fan_in)  # He initialisation, suits ReLU
        self.W = rng.normal(
            0.0, scale, size=(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.b = np.zeros((out_channels,))

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._cols = None  # cached im2col matrix for the backward pass

    def forward(self, x, training=False):
        self.input = x
        n, c, h, w = x.shape
        out_h = _conv_out_size(h, self.k, self.stride, self.pad)
        out_w = _conv_out_size(w, self.k, self.stride, self.pad)

        cols = im2col(x, self.k, self.k, self.stride, self.pad)
        w_col = self.W.reshape(self.out_channels, -1).T  # (C*k*k, out_channels)

        out = cols @ w_col + self.b  # (N*out_h*out_w, out_channels)
        out = out.reshape(n, out_h, out_w, self.out_channels).transpose(0, 3, 1, 2)

        self._cols = cols
        self.output = out
        return out

    def backward(self, grad_output):
        # grad_output: (N, out_channels, out_h, out_w)
        grad = grad_output.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)

        self.db = np.sum(grad, axis=0)
        dw_col = self._cols.T @ grad  # (C*k*k, out_channels)
        self.dW = dw_col.T.reshape(self.W.shape)

        w_col = self.W.reshape(self.out_channels, -1)  # (out_channels, C*k*k)
        d_cols = grad @ w_col  # (N*out_h*out_w, C*k*k)
        return col2im(d_cols, self.input.shape, self.k, self.k, self.stride, self.pad)

    def params_and_grads(self):
        yield self.W, self.dW
        yield self.b, self.db


class MaxPool2D(Layer):
    """Non-overlapping max pooling (stride equals the pool size).

    Requires the spatial dimensions to be divisible by ``pool_size``.
    """

    def __init__(self, pool_size=2):
        super().__init__()
        self.p = pool_size
        self._mask = None
        self._shape = None

    def forward(self, x, training=False):
        n, c, h, w = x.shape
        p = self.p
        if h % p or w % p:
            raise ValueError(
                f"MaxPool2D: input {h}x{w} not divisible by pool size {p}"
            )
        self.input = x
        self._shape = x.shape

        # Tile into (N, C, out_h, p, out_w, p) and reduce over the two pool axes.
        x_r = x.reshape(n, c, h // p, p, w // p, p)
        out = x_r.max(axis=(3, 5))

        # Remember which element won, to route gradients in backward.
        self._mask = (x_r == out[:, :, :, None, :, None])
        self.output = out
        return out

    def backward(self, grad_output):
        n, c, h, w = self._shape
        p = self.p
        # Broadcast the upstream grad across each pooling window, gated by mask.
        grad = grad_output[:, :, :, None, :, None]
        dx = self._mask * grad
        return dx.reshape(n, c, h, w)


class Flatten(Layer):
    """Reshape ``(N, C, H, W)`` -> ``(N, C*H*W)`` and back on the way down."""

    def forward(self, x, training=False):
        self.input = x
        self._shape = x.shape
        self.output = x.reshape(x.shape[0], -1)
        return self.output

    def backward(self, grad_output):
        return grad_output.reshape(self._shape)

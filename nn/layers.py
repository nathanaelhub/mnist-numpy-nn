"""Layers: the building blocks of a network.

Every layer implements two methods:

``forward(x)``   -- consume an input, cache whatever the backward pass needs,
                    and return the output activation.
``backward(grad)`` -- given dL/d(output), return dL/d(input). Parameterised
                    layers (e.g. :class:`Dense`) additionally stash dL/d(param)
                    on themselves so an optimizer can read it later.

Inputs flow through as 2-D arrays shaped ``(batch_size, features)``.
"""

import numpy as np


class Layer:
    """Abstract base class for all layers."""

    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, x, training=False):
        """Compute the layer output.

        ``training`` distinguishes the train-time pass (used by stochastic
        layers such as :class:`Dropout`) from inference. Deterministic layers
        ignore it.
        """
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError

    def params_and_grads(self):
        """Yield ``(param, grad)`` pairs for trainable layers.

        Both items are references to the *same* arrays the layer uses, so an
        optimizer can update parameters in place. The default layer has none.
        """
        return iter(())


class Dense(Layer):
    """A fully connected (affine) layer: ``y = x @ W + b``.

    Parameters
    ----------
    in_features, out_features : int
        Sizes of the input and output feature vectors.
    weight_init : {"he", "xavier"}
        Scaling scheme for the initial random weights. "he" suits ReLU,
        "xavier" suits sigmoid/tanh.
    """

    def __init__(self, in_features, out_features, weight_init="he", seed=None):
        super().__init__()
        rng = np.random.default_rng(seed)

        if weight_init == "he":
            scale = np.sqrt(2.0 / in_features)
        elif weight_init == "xavier":
            scale = np.sqrt(1.0 / in_features)
        else:
            raise ValueError(f"unknown weight_init: {weight_init!r}")

        self.W = rng.normal(0.0, scale, size=(in_features, out_features))
        self.b = np.zeros((1, out_features))

        # Gradients, filled in by ``backward``.
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x, training=False):
        self.input = x
        self.output = x @ self.W + self.b
        return self.output

    def backward(self, grad_output):
        # grad_output: dL/dy with shape (batch, out_features)
        self.dW = self.input.T @ grad_output
        self.db = np.sum(grad_output, axis=0, keepdims=True)
        return grad_output @ self.W.T  # dL/dx

    def params_and_grads(self):
        yield self.W, self.dW
        yield self.b, self.db


class Activation(Layer):
    """Element-wise (or row-wise, for softmax) non-linearity.

    A single configurable class covers the three activations the framework
    supports. Choose one by name::

        Activation("relu")
        Activation("sigmoid")
        Activation("softmax")
    """

    SUPPORTED = ("sigmoid", "relu", "softmax")

    def __init__(self, name):
        super().__init__()
        name = name.lower()
        if name not in self.SUPPORTED:
            raise ValueError(
                f"unknown activation {name!r}; choose from {self.SUPPORTED}"
            )
        self.name = name

    # -- forward ---------------------------------------------------------
    def forward(self, x, training=False):
        self.input = x
        if self.name == "relu":
            self.output = np.maximum(0.0, x)
        elif self.name == "sigmoid":
            # Numerically stable logistic for both signs of x.
            self.output = np.where(
                x >= 0,
                1.0 / (1.0 + np.exp(-np.clip(x, -500, 500))),
                np.exp(np.clip(x, -500, 500)) / (1.0 + np.exp(np.clip(x, -500, 500))),
            )
        else:  # softmax, computed row-wise with the max-subtraction trick
            shifted = x - np.max(x, axis=1, keepdims=True)
            exp = np.exp(shifted)
            self.output = exp / np.sum(exp, axis=1, keepdims=True)
        return self.output

    # -- backward --------------------------------------------------------
    def backward(self, grad_output):
        s = self.output
        if self.name == "relu":
            return grad_output * (self.input > 0)
        if self.name == "sigmoid":
            return grad_output * s * (1.0 - s)

        # Softmax: full Jacobian-vector product, computed per row.
        #   dL/dz_i = s_i * (g_i - sum_j g_j s_j)
        # This is exact for *any* upstream gradient, so it composes correctly
        # with cross-entropy or anything else.
        dot = np.sum(grad_output * s, axis=1, keepdims=True)
        return s * (grad_output - dot)


class Dropout(Layer):
    """Inverted dropout regularisation.

    During training each activation is kept with probability ``1 - rate`` and
    the survivors are scaled by ``1 / (1 - rate)`` so the expected output is
    unchanged. At inference time the layer is the identity, so no rescaling is
    needed elsewhere.

    Parameters
    ----------
    rate : float
        Fraction of activations to zero out, in ``[0, 1)``.
    seed : int, optional
        Seed for the internal RNG, for reproducible masks.
    """

    def __init__(self, rate=0.5, seed=None):
        super().__init__()
        if not 0.0 <= rate < 1.0:
            raise ValueError(f"dropout rate must be in [0, 1), got {rate}")
        self.rate = rate
        self.keep = 1.0 - rate
        self.mask = None
        self._rng = np.random.default_rng(seed)

    def forward(self, x, training=False):
        self.input = x
        if not training or self.rate == 0.0:
            self.mask = None
            self.output = x
            return x
        # Inverted dropout: sample a keep-mask and pre-scale by 1/keep.
        self.mask = (self._rng.random(x.shape) < self.keep) / self.keep
        self.output = x * self.mask
        return self.output

    def backward(self, grad_output):
        # Identity pass (eval, or rate 0) stored no mask.
        if self.mask is None:
            return grad_output
        return grad_output * self.mask

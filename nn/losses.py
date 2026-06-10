"""Loss functions.

Each loss implements ``forward(y_pred, y_true) -> scalar`` and
``backward(y_pred, y_true) -> dL/dy_pred``. Gradients are averaged over the
batch so the learning rate is independent of batch size.
"""

import numpy as np

_EPS = 1e-12  # guards against log(0) and divide-by-zero


class Loss:
    """Abstract base class for losses."""

    def forward(self, y_pred, y_true):
        raise NotImplementedError

    def backward(self, y_pred, y_true):
        raise NotImplementedError

    # Allow ``loss(y_pred, y_true)`` as shorthand for the forward value.
    def __call__(self, y_pred, y_true):
        return self.forward(y_pred, y_true)


class MSE(Loss):
    """Mean squared error, averaged over the whole batch."""

    def forward(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

    def backward(self, y_pred, y_true):
        # d/dy_pred of mean((y_pred - y_true)^2)
        n = y_pred.shape[0] * y_pred.shape[1]
        return 2.0 * (y_pred - y_true) / n


class CrossEntropy(Loss):
    """Categorical cross-entropy for one-hot targets.

    Expects ``y_pred`` to already be a probability distribution per row (i.e.
    the output of a softmax layer). The gradient returned is dL/d(probabilities);
    paired with :class:`~nn.layers.Activation` softmax it reduces cleanly to
    ``(p - y) / N``.
    """

    def forward(self, y_pred, y_true):
        n = y_pred.shape[0]
        clipped = np.clip(y_pred, _EPS, 1.0)
        return -np.sum(y_true * np.log(clipped)) / n

    def backward(self, y_pred, y_true):
        n = y_pred.shape[0]
        clipped = np.clip(y_pred, _EPS, 1.0)
        return -(y_true / clipped) / n

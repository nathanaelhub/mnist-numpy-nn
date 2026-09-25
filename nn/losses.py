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

    @staticmethod
    def from_logits(logits, y_true):
        """Exact loss and dL/d(logits) for softmax + cross-entropy, fused.

        Going through probabilities loses the gradient exactly when it matters:
        for a confidently *wrong* prediction p_target underflows, the clip in
        :meth:`backward` caps 1/p at 1e12, and the softmax Jacobian then
        multiplies by p ~ 1e-22 — a gradient of ~1e-10 where the true one is
        ~1. Working from the logits avoids both the clip and the cancellation:

            loss = mean_rows( logsumexp(z) - sum(y * z) )
            dL/dz = (softmax(z) - y) / N

        (Exact for rows of ``y_true`` that sum to 1, i.e. one-hot or soft labels.)
        """
        n = logits.shape[0]
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        log_sum = np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
        log_probs = shifted - log_sum
        loss = -np.sum(y_true * log_probs) / n
        grad = (np.exp(log_probs) - y_true) / n
        return loss, grad

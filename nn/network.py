"""The :class:`Network` container: stack layers, compile, fit, predict."""

import numpy as np

from .progress import ProgressBar


class Network:
    """A sequential stack of layers trained by mini-batch gradient descent.

    Typical use::

        net = Network()
        net.add(Dense(784, 128)); net.add(Activation("relu"))
        net.add(Dense(128, 10));  net.add(Activation("softmax"))
        net.compile(loss=CrossEntropy(), optimizer=Adam(lr=1e-3))
        net.fit(X_train, y_train, epochs=10, batch_size=64,
                validation_data=(X_val, y_val))
    """

    def __init__(self):
        self.layers = []
        self.loss = None
        self.optimizer = None

    # -- construction ----------------------------------------------------
    def add(self, layer):
        """Append a layer to the stack. Returns ``self`` for chaining."""
        self.layers.append(layer)
        return self

    def compile(self, loss, optimizer):
        """Attach a loss function and an optimizer."""
        self.loss = loss
        self.optimizer = optimizer
        return self

    # -- forward / backward ---------------------------------------------
    def forward(self, x, training=False):
        for layer in self.layers:
            x = layer.forward(x, training=training)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    # -- inference -------------------------------------------------------
    def predict(self, x):
        """Return raw network outputs (probabilities for a softmax head)."""
        return self.forward(x)

    def predict_classes(self, x):
        return np.argmax(self.forward(x), axis=1)

    def evaluate(self, x, y):
        """Return ``(loss, accuracy)`` over ``x``/``y`` in one forward pass."""
        preds = self.forward(x)
        loss = self.loss.forward(preds, y)
        acc = self._accuracy(preds, y)
        return loss, acc

    @staticmethod
    def _accuracy(preds, y):
        return float(np.mean(np.argmax(preds, axis=1) == np.argmax(y, axis=1)))

    # -- training --------------------------------------------------------
    def fit(self, X, y, epochs, batch_size, validation_data=None,
            shuffle=True, verbose=True, seed=None):
        """Train with mini-batches and a live per-epoch progress bar.

        Returns a history dict of per-epoch metric lists.
        """
        if self.loss is None or self.optimizer is None:
            raise RuntimeError("call compile() before fit()")

        n = X.shape[0]
        rng = np.random.default_rng(seed)
        history = {"loss": [], "acc": []}
        if validation_data is not None:
            history["val_loss"] = []
            history["val_acc"] = []

        for epoch in range(1, epochs + 1):
            order = rng.permutation(n) if shuffle else np.arange(n)

            if verbose:
                print(f"Epoch {epoch}/{epochs}")
                bar = ProgressBar(total=n, prefix="")

            running_loss = 0.0
            running_correct = 0
            seen = 0

            for start in range(0, n, batch_size):
                idx = order[start:start + batch_size]
                xb, yb = X[idx], y[idx]

                preds = self.forward(xb, training=True)
                batch_loss = self.loss.forward(preds, yb)
                grad = self.loss.backward(preds, yb)
                self.backward(grad)
                self.optimizer.step(self.layers)

                bs = xb.shape[0]
                seen += bs
                running_loss += batch_loss * bs
                running_correct += int(
                    np.sum(np.argmax(preds, axis=1) == np.argmax(yb, axis=1))
                )

                if verbose:
                    bar.update(seen, {
                        "loss": running_loss / seen,
                        "acc": running_correct / seen,
                    })

            epoch_loss = running_loss / seen
            epoch_acc = running_correct / seen
            history["loss"].append(epoch_loss)
            history["acc"].append(epoch_acc)

            if validation_data is not None:
                val_loss, val_acc = self.evaluate(*validation_data)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)
                if verbose:
                    bar.update(seen, {
                        "loss": epoch_loss, "acc": epoch_acc,
                        "val_loss": val_loss, "val_acc": val_acc,
                    })

            if verbose:
                bar.close()

        return history

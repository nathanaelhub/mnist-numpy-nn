"""Train an MLP and save visual diagnostics to ``assets/``.

Produces two figures:

* ``training_curves.png`` -- loss and accuracy over epochs (train vs. validation)
* ``misclassified.png``   -- a grid of test digits the model got wrong, each
  captioned with its predicted and true label.

Usage::

    pip install matplotlib       # one-off, only needed for plotting
    python plot_results.py
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend: write files without a display
import matplotlib.pyplot as plt

from data import load_data
from nn import Network, Dense, Activation, Adam, CrossEntropy

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def build_and_train(epochs, batch_size, seed):
    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = load_data(seed=seed)

    net = Network()
    net.add(Dense(784, 128, seed=seed)); net.add(Activation("relu"))
    net.add(Dense(128, 64, seed=seed + 1)); net.add(Activation("relu"))
    net.add(Dense(64, 10, weight_init="xavier", seed=seed + 2))
    net.add(Activation("softmax"))
    net.compile(CrossEntropy(), Adam(1e-3))

    history = net.fit(
        X_tr, y_tr, epochs=epochs, batch_size=batch_size,
        validation_data=(X_val, y_val), seed=seed,
    )
    return net, history, (X_te, y_te)


def plot_curves(history, path):
    epochs = range(1, len(history["loss"]) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))

    ax_loss.plot(epochs, history["loss"], "o-", label="train")
    ax_loss.plot(epochs, history["val_loss"], "s-", label="validation")
    ax_loss.set(title="Cross-entropy loss", xlabel="epoch", ylabel="loss")
    ax_loss.legend(); ax_loss.grid(alpha=0.3)

    ax_acc.plot(epochs, history["acc"], "o-", label="train")
    ax_acc.plot(epochs, history["val_acc"], "s-", label="validation")
    ax_acc.set(title="Accuracy", xlabel="epoch", ylabel="accuracy")
    ax_acc.legend(); ax_acc.grid(alpha=0.3)

    fig.suptitle("MNIST MLP (784-128-64-10), Adam", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_misclassified(net, X_te, y_te, path, n=15):
    preds = net.predict_classes(X_te)
    truth = np.argmax(y_te, axis=1)
    wrong = np.where(preds != truth)[0][:n]

    cols = 5
    rows = (len(wrong) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.8))
    for ax, idx in zip(axes.ravel(), wrong):
        ax.imshow(X_te[idx].reshape(28, 28), cmap="gray")
        ax.set_title(f"pred {preds[idx]} / true {truth[idx]}", fontsize=8)
        ax.axis("off")
    for ax in axes.ravel()[len(wrong):]:
        ax.axis("off")

    fig.suptitle("Misclassified test digits", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Plot MNIST training diagnostics.")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    os.makedirs(ASSETS_DIR, exist_ok=True)
    net, history, (X_te, y_te) = build_and_train(
        args.epochs, args.batch_size, args.seed
    )

    curves = os.path.join(ASSETS_DIR, "training_curves.png")
    grid = os.path.join(ASSETS_DIR, "misclassified.png")
    plot_curves(history, curves)
    plot_misclassified(net, X_te, y_te, grid)

    test_acc = net.evaluate(X_te, y_te)[1]
    print(f"\nTest accuracy: {test_acc * 100:.2f}%")
    print(f"Saved {curves}")
    print(f"Saved {grid}")


if __name__ == "__main__":
    main()

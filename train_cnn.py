"""Train a small convolutional network on MNIST.

The convolutions are pure-NumPy im2col, so this is slower than ``train.py``.
By default it trains on a subset to stay snappy; pass ``--full`` for the whole
60k training set, which reaches ~99% test accuracy given enough epochs.

Examples
--------
    python train_cnn.py                       # quick demo on a 8k subset
    python train_cnn.py --full --epochs 5     # full data, higher accuracy
"""

import argparse

import numpy as np

from data import load_data
from nn import (
    Network, Conv2D, MaxPool2D, Flatten, Dense, Activation, Adam, CrossEntropy,
)


def build_cnn(seed=0):
    """LeNet-style stack: two conv/pool blocks, then a small classifier head."""
    net = Network()
    net.add(Conv2D(1, 8, kernel_size=3, padding=1, seed=seed))   # 28x28 -> 28x28
    net.add(Activation("relu"))
    net.add(MaxPool2D(2))                                         # -> 14x14
    net.add(Conv2D(8, 16, kernel_size=3, padding=1, seed=seed + 1))  # -> 14x14
    net.add(Activation("relu"))
    net.add(MaxPool2D(2))                                         # -> 7x7
    net.add(Flatten())                                           # 16*7*7 = 784
    net.add(Dense(16 * 7 * 7, 64, seed=seed + 2))
    net.add(Activation("relu"))
    net.add(Dense(64, 10, weight_init="xavier", seed=seed + 3))
    net.add(Activation("softmax"))
    return net


def parse_args():
    p = argparse.ArgumentParser(description="Train a NumPy CNN on MNIST.")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--subset", type=int, default=8000,
                   help="train on this many images (ignored with --full)")
    p.add_argument("--full", action="store_true",
                   help="use the entire 60k training set")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    print("Loading MNIST (as 28x28 image tensors)...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data(
        flatten=False, seed=args.seed
    )

    if not args.full:
        X_train, y_train = X_train[:args.subset], y_train[:args.subset]
        X_val, y_val = X_val[:args.subset // 5], y_val[:args.subset // 5]
        print(f"  (subset) train {X_train.shape[0]}  val {X_val.shape[0]}")
    else:
        print(f"  (full) train {X_train.shape[0]}  val {X_val.shape[0]}")
    print(f"  input shape {X_train.shape[1:]}  test {X_test.shape[0]}\n")

    net = build_cnn(seed=args.seed)
    net.compile(loss=CrossEntropy(), optimizer=Adam(lr=args.lr))

    print("Architecture: Conv(1->8) -> Pool -> Conv(8->16) -> Pool "
          "-> Flatten -> Dense(784->64) -> Dense(64->10)\n")

    net.fit(
        X_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_val, y_val),
        seed=args.seed,
    )

    test_loss, test_acc = net.evaluate(X_test, y_test)
    print(f"\nTest set:  loss {test_loss:.4f}  accuracy {test_acc * 100:.2f}%")


if __name__ == "__main__":
    main()

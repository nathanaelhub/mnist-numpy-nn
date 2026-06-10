"""Train a small fully connected network on MNIST.

Examples
--------
    python train.py                       # sensible defaults (Adam, 10 epochs)
    python train.py --optimizer sgd --lr 0.1 --epochs 20
    python train.py --hidden 256 128 --batch-size 128
"""

import argparse

import numpy as np

from data import load_data
from nn import Network, Dense, Activation, Dropout, SGD, Adam, MSE, CrossEntropy


def build_network(input_dim, hidden_sizes, num_classes, output_activation,
                  seed, dropout=0.0):
    """Stack Dense + ReLU (+ optional Dropout) blocks, then a head."""
    net = Network()
    prev = input_dim
    for h in hidden_sizes:
        net.add(Dense(prev, h, weight_init="he", seed=seed))
        net.add(Activation("relu"))
        if dropout > 0.0:
            net.add(Dropout(dropout, seed=seed))
        prev = h
    # Output head: softmax for cross-entropy, sigmoid for MSE.
    net.add(Dense(prev, num_classes, weight_init="xavier", seed=seed))
    net.add(Activation(output_activation))
    return net


def parse_args():
    p = argparse.ArgumentParser(description="Train a NumPy MLP on MNIST.")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, nargs="+", default=[128])
    p.add_argument("--dropout", type=float, default=0.0,
                   help="dropout rate after each hidden layer (0 disables)")
    p.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    p.add_argument("--momentum", type=float, default=0.9,
                   help="momentum for SGD (ignored by Adam)")
    p.add_argument("--loss", choices=["cross_entropy", "mse"],
                   default="cross_entropy")
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    print("Loading MNIST...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data(
        val_split=args.val_split, seed=args.seed
    )
    print(f"  train {X_train.shape[0]}  val {X_val.shape[0]}  test {X_test.shape[0]}\n")

    # Loss determines the appropriate output activation.
    if args.loss == "cross_entropy":
        loss, out_act = CrossEntropy(), "softmax"
    else:
        loss, out_act = MSE(), "sigmoid"

    net = build_network(
        input_dim=X_train.shape[1],
        hidden_sizes=args.hidden,
        num_classes=10,
        output_activation=out_act,
        seed=args.seed,
        dropout=args.dropout,
    )

    if args.optimizer == "adam":
        optimizer = Adam(lr=args.lr)
    else:
        optimizer = SGD(lr=args.lr, momentum=args.momentum)

    net.compile(loss=loss, optimizer=optimizer)

    arch = " -> ".join(["784"] + [str(h) for h in args.hidden] + ["10"])
    print(f"Architecture: {arch}")
    print(f"Optimizer: {args.optimizer}  lr={args.lr}  loss={args.loss}\n")

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

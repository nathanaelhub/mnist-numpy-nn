# mnist_nn — a NumPy-only neural network framework

A small, modular, object-oriented neural network library built with **nothing but
NumPy**, plus a training script that classifies the MNIST handwritten digits.

A 2-layer MLP reaches **~95% test accuracy in 2 epochs** (a few seconds on CPU)
and ~98% with a slightly bigger net and more epochs.

## Layout

```
mnist_nn/
├── nn/
│   ├── layers.py       # Layer base class; Dense, Activation, Dropout
│   ├── losses.py       # Loss base class; MSE and CrossEntropy
│   ├── optimizers.py   # Optimizer base class; SGD (+momentum) and Adam
│   ├── network.py      # Network: add / compile / fit / predict / evaluate
│   └── progress.py     # dependency-free terminal progress bar
├── data/
│   └── mnist_loader.py # downloads + caches mnist.npz, builds train/val/test
├── tests/
│   └── test_nn.py      # finite-difference gradient checks + behaviour tests
├── train.py            # CLI entry point
└── README.md
```

## Quick start

```bash
python train.py                                  # Adam, 1 hidden layer, 10 epochs
python train.py --hidden 256 128 --epochs 20     # deeper net
python train.py --optimizer sgd --lr 0.1         # SGD + momentum
python train.py --loss mse                       # MSE head (sigmoid output)
python train.py --hidden 256 --dropout 0.3       # dropout regularisation
```

The first run downloads MNIST (~11 MB) to `data/mnist.npz` and reuses it afterward.
Pre-download without training with `python -m data.mnist_loader`.

## Design

Every **layer** implements `forward(x, training=False)` and `backward(grad)`. The
forward pass caches whatever the backward pass needs; the backward pass returns
`dL/d(input)` and, for parameterised layers, stores `dL/d(param)` on the layer.
An **optimizer** then walks each layer's `params_and_grads()` and updates the
arrays in place. The `training` flag lets stochastic layers behave differently
at train vs. inference time — `fit` passes `training=True`, while `predict` /
`evaluate` pass `training=False`.

- **`Dense`** — affine `y = x·W + b`, with He or Xavier initialisation.
- **`Activation`** — one configurable class for `"relu"`, `"sigmoid"`, `"softmax"`.
  Softmax uses the exact Jacobian-vector product in its backward pass, so it
  composes correctly with *any* loss (not just cross-entropy).
- **`Dropout`** — inverted dropout: zeros a fraction of activations and rescales
  the survivors during training; the identity at inference.
- **`CrossEntropy`** — for one-hot targets over softmax probabilities. Combined
  with the softmax layer the gradient reduces cleanly to `(p − y) / N`.
- **`MSE`** — mean squared error, batch-averaged.
- **`SGD`** — with optional classical momentum. **`Adam`** — bias-corrected.

Correctness is checked against finite-difference gradients (agreement to ~1e-12).

## Tests

```bash
pip install pytest
python -m pytest          # 22 tests: per-layer + whole-network gradient checks,
                          # activation/loss values, dropout behaviour, learning
```

Every layer and loss is verified by comparing its analytic gradient to a
central finite-difference estimate of the same scalar loss.

## Library use

```python
import numpy as np
from nn import Network, Dense, Activation, Adam, CrossEntropy
from data import load_data

(X_tr, y_tr), (X_val, y_val), (X_te, y_te) = load_data()

net = Network()
net.add(Dense(784, 128)); net.add(Activation("relu"))
net.add(Dense(128, 10));  net.add(Activation("softmax"))
net.compile(loss=CrossEntropy(), optimizer=Adam(lr=1e-3))

net.fit(X_tr, y_tr, epochs=10, batch_size=64, validation_data=(X_val, y_val))
print("test accuracy:", net.evaluate(X_te, y_te)[1])
```

## Requirements

Python 3.8+ and NumPy. No other dependencies.
```bash
pip install numpy
```

"""Correctness tests for the NumPy neural-network framework.

The backbone is a finite-difference gradient check: for a scalar loss L and a
parameter array p, the analytic gradient dL/dp produced by ``backward`` must
match the central-difference estimate ``(L(p+e) - L(p-e)) / 2e`` everywhere.

Run from the project root::

    python -m pytest -q
"""

import numpy as np
import pytest

from nn import (
    Network, Dense, Activation, Dropout, Conv2D, MaxPool2D, Flatten,
    SGD, Adam, MSE, CrossEntropy,
)

EPS = 1e-5
ATOL = 1e-6


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def numeric_grad(scalar_fn, x):
    """Central-difference gradient of ``scalar_fn()`` w.r.t. array ``x``.

    ``scalar_fn`` must read the *current* contents of ``x`` (a closure), so
    mutating ``x`` in place between calls changes the result.
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + EPS
        plus = scalar_fn()
        x[idx] = orig - EPS
        minus = scalar_fn()
        x[idx] = orig
        grad[idx] = (plus - minus) / (2 * EPS)
        it.iternext()
    return grad


def one_hot(idx, n_classes):
    out = np.zeros((idx.shape[0], n_classes))
    out[np.arange(idx.shape[0]), idx] = 1.0
    return out


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


# --------------------------------------------------------------------------
# Dense layer
# --------------------------------------------------------------------------
def test_dense_forward_shape_and_value(rng):
    d = Dense(4, 3, seed=0)
    x = rng.normal(size=(5, 4))
    out = d.forward(x)
    assert out.shape == (5, 3)
    np.testing.assert_allclose(out, x @ d.W + d.b)


def test_dense_gradients(rng):
    """dW, db and dInput all match finite differences (through softmax+CE)."""
    d = Dense(4, 3, seed=0)
    x = rng.normal(size=(6, 4))
    y = one_hot(rng.integers(0, 3, size=6), 3)
    sm, ce = Activation("softmax"), CrossEntropy()

    def loss():
        return ce.forward(sm.forward(d.forward(x)), y)

    loss()
    grad_in = d.backward(sm.backward(ce.backward(sm.output, y)))

    np.testing.assert_allclose(numeric_grad(loss, d.W), d.dW, atol=ATOL)
    np.testing.assert_allclose(numeric_grad(loss, d.b), d.db, atol=ATOL)
    np.testing.assert_allclose(numeric_grad(loss, x), grad_in, atol=ATOL)


def test_dense_init_schemes():
    he = Dense(100, 100, weight_init="he", seed=0)
    xa = Dense(100, 100, weight_init="xavier", seed=0)
    # He variance (~2/n) should exceed Xavier variance (~1/n).
    assert he.W.var() > xa.W.var()
    with pytest.raises(ValueError):
        Dense(4, 3, weight_init="bogus")


# --------------------------------------------------------------------------
# Activations
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["sigmoid", "relu", "softmax"])
def test_activation_gradient(name, rng):
    """backward(g) equals the numeric gradient of sum(g * forward(x))."""
    act = Activation(name)
    # Keep ReLU inputs away from the non-differentiable kink at 0.
    x = rng.normal(size=(5, 4))
    if name == "relu":
        x = np.sign(x) * (np.abs(x) + 0.5)
    g = rng.normal(size=(5, 4))

    def scalar():
        return np.sum(g * act.forward(x))

    scalar()
    analytic = act.backward(g)
    np.testing.assert_allclose(numeric_grad(scalar, x), analytic, atol=ATOL)


def test_softmax_is_a_distribution(rng):
    sm = Activation("softmax")
    out = sm.forward(rng.normal(size=(7, 5)))
    np.testing.assert_allclose(out.sum(axis=1), np.ones(7), atol=1e-12)
    assert np.all(out > 0)


def test_softmax_numerically_stable():
    sm = Activation("softmax")
    out = sm.forward(np.array([[1000.0, 1001.0, 1002.0]]))
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out.sum(), 1.0, atol=1e-12)


def test_sigmoid_handles_extremes():
    sig = Activation("sigmoid")
    out = sig.forward(np.array([[-1000.0, 0.0, 1000.0]]))
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out, [[0.0, 0.5, 1.0]], atol=1e-9)


def test_unknown_activation_raises():
    with pytest.raises(ValueError):
        Activation("tanh")


# --------------------------------------------------------------------------
# Losses
# --------------------------------------------------------------------------
def test_mse_value_and_gradient(rng):
    mse = MSE()
    p = rng.normal(size=(4, 3))
    y = rng.normal(size=(4, 3))
    np.testing.assert_allclose(mse.forward(p, y), np.mean((p - y) ** 2))

    def loss():
        return mse.forward(p, y)

    np.testing.assert_allclose(numeric_grad(loss, p), mse.backward(p, y), atol=ATOL)


def test_cross_entropy_value_and_gradient(rng):
    ce = CrossEntropy()
    p = Activation("softmax").forward(rng.normal(size=(4, 3)))  # valid probs
    y = one_hot(rng.integers(0, 3, size=4), 3)
    expected = -np.sum(y * np.log(p)) / 4
    np.testing.assert_allclose(ce.forward(p, y), expected)

    def loss():
        return ce.forward(p, y)

    np.testing.assert_allclose(numeric_grad(loss, p), ce.backward(p, y), atol=1e-5)


# --------------------------------------------------------------------------
# Dropout
# --------------------------------------------------------------------------
def test_dropout_is_identity_in_eval(rng):
    drop = Dropout(0.5, seed=0)
    x = rng.normal(size=(8, 6))
    out = drop.forward(x, training=False)
    np.testing.assert_array_equal(out, x)
    g = rng.normal(size=(8, 6))
    np.testing.assert_array_equal(drop.backward(g), g)


def test_dropout_rate_zero_is_identity(rng):
    drop = Dropout(0.0, seed=0)
    x = rng.normal(size=(8, 6))
    np.testing.assert_array_equal(drop.forward(x, training=True), x)


def test_dropout_mask_and_scaling(rng):
    rate = 0.4
    drop = Dropout(rate, seed=0)
    x = np.ones((1000, 1000))
    out = drop.forward(x, training=True)

    # Surviving units are scaled by 1/keep; the rest are zero.
    kept = out > 0
    np.testing.assert_allclose(out[kept], 1.0 / (1.0 - rate))
    # Empirical keep fraction is close to (1 - rate).
    assert abs(kept.mean() - (1.0 - rate)) < 0.01
    # Expected value is preserved.
    assert abs(out.mean() - 1.0) < 0.01


def test_dropout_backward_uses_mask(rng):
    drop = Dropout(0.5, seed=3)
    x = rng.normal(size=(8, 6))
    drop.forward(x, training=True)
    g = rng.normal(size=(8, 6))
    np.testing.assert_array_equal(drop.backward(g), g * drop.mask)


def test_dropout_invalid_rate():
    with pytest.raises(ValueError):
        Dropout(1.0)


# --------------------------------------------------------------------------
# Convolutional layers
# --------------------------------------------------------------------------
def test_conv2d_forward_shape(rng):
    conv = Conv2D(3, 4, kernel_size=3, stride=1, padding=1, seed=0)
    x = rng.normal(size=(2, 3, 8, 8))
    out = conv.forward(x)
    assert out.shape == (2, 4, 8, 8)  # "same" padding keeps H, W


def test_conv2d_stride_and_padding_shape(rng):
    conv = Conv2D(1, 2, kernel_size=3, stride=2, padding=0, seed=0)
    out = conv.forward(rng.normal(size=(2, 1, 9, 9)))
    assert out.shape == (2, 2, 4, 4)  # (9 - 3)//2 + 1 = 4


def test_conv2d_gradients(rng):
    """dW, db and dInput match finite differences (small tensor)."""
    conv = Conv2D(2, 3, kernel_size=3, stride=1, padding=1, seed=0)
    x = rng.normal(size=(2, 2, 5, 5))
    # A fixed random upstream gradient; check the scalar sum(g * out).
    g = rng.normal(size=(2, 3, 5, 5))

    def scalar():
        return np.sum(g * conv.forward(x))

    scalar()
    grad_in = conv.backward(g)

    np.testing.assert_allclose(numeric_grad(scalar, conv.W), conv.dW, atol=ATOL)
    np.testing.assert_allclose(numeric_grad(scalar, conv.b), conv.db, atol=ATOL)
    np.testing.assert_allclose(numeric_grad(scalar, x), grad_in, atol=ATOL)


def test_maxpool_forward_and_gradient(rng):
    pool = MaxPool2D(2)
    x = rng.normal(size=(2, 3, 4, 4))
    out = pool.forward(x)
    assert out.shape == (2, 3, 2, 2)

    g = rng.normal(size=(2, 3, 2, 2))

    def scalar():
        return np.sum(g * pool.forward(x))

    scalar()
    grad_in = pool.backward(g)
    np.testing.assert_allclose(numeric_grad(scalar, x), grad_in, atol=ATOL)


def test_maxpool_rejects_indivisible():
    with pytest.raises(ValueError):
        MaxPool2D(2).forward(np.zeros((1, 1, 5, 5)))


def test_flatten_roundtrip(rng):
    flat = Flatten()
    x = rng.normal(size=(4, 2, 3, 3))
    out = flat.forward(x)
    assert out.shape == (4, 18)
    # backward restores the original shape exactly.
    np.testing.assert_array_equal(flat.backward(out), x)


def test_conv_net_learns(rng):
    """A tiny CNN should drive its loss down on a small batch."""
    x = rng.normal(size=(16, 1, 8, 8))
    y = one_hot(rng.integers(0, 3, size=16), 3)

    net = Network()
    net.add(Conv2D(1, 4, kernel_size=3, padding=1, seed=0))
    net.add(Activation("relu"))
    net.add(MaxPool2D(2))                 # 8x8 -> 4x4
    net.add(Flatten())                    # 4 * 4 * 4 = 64
    net.add(Dense(64, 3, seed=1))
    net.add(Activation("softmax"))
    net.compile(CrossEntropy(), Adam(1e-2))

    hist = net.fit(x, y, epochs=20, batch_size=8, verbose=False, seed=0)
    assert hist["loss"][-1] < hist["loss"][0]


# --------------------------------------------------------------------------
# Whole-network gradient check
# --------------------------------------------------------------------------
def _build_small_net(out_act, seed=0):
    net = Network()
    net.add(Dense(6, 5, seed=seed))
    net.add(Activation("relu"))
    net.add(Dense(5, 3, seed=seed + 1))
    net.add(Activation(out_act))
    return net


@pytest.mark.parametrize("out_act,loss_cls", [
    ("softmax", CrossEntropy),
    ("sigmoid", MSE),
])
def test_full_network_param_gradients(out_act, loss_cls, rng):
    """Every Dense parameter's analytic grad matches finite differences."""
    net = _build_small_net(out_act)
    loss = loss_cls()
    x = rng.normal(size=(4, 6))
    y = one_hot(rng.integers(0, 3, size=4), 3)

    def total_loss():
        return loss.forward(net.forward(x, training=False), y)

    preds = net.forward(x, training=False)
    net.backward(loss.backward(preds, y))

    for layer in net.layers:
        for param, grad in layer.params_and_grads():
            np.testing.assert_allclose(
                numeric_grad(total_loss, param), grad, atol=ATOL,
            )


# --------------------------------------------------------------------------
# End-to-end learning
# --------------------------------------------------------------------------
@pytest.mark.parametrize("optimizer", [Adam(1e-2), SGD(0.5, momentum=0.9)])
def test_network_learns_synthetic(optimizer, rng):
    """A linearly-separable problem should be fit to high accuracy."""
    n, d, c = 400, 10, 3
    X = rng.normal(size=(n, d))
    W = rng.normal(size=(d, c))
    Y = one_hot(np.argmax(X @ W, axis=1), c)

    net = Network()
    net.add(Dense(d, 16, seed=0)); net.add(Activation("relu"))
    net.add(Dense(16, c, seed=1)); net.add(Activation("softmax"))
    net.compile(CrossEntropy(), optimizer)

    hist = net.fit(X, Y, epochs=40, batch_size=32, verbose=False, seed=0)
    assert hist["loss"][-1] < hist["loss"][0]   # loss decreased
    assert hist["acc"][-1] > 0.9                # learned the task


def test_compile_required_before_fit(rng):
    net = Network()
    net.add(Dense(3, 2)); net.add(Activation("softmax"))
    with pytest.raises(RuntimeError):
        net.fit(rng.normal(size=(4, 3)), one_hot(rng.integers(0, 2, 4), 2),
                epochs=1, batch_size=2, verbose=False)

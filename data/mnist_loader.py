"""Download, cache, and prepare the MNIST handwritten-digit dataset.

The data is fetched once as a single ``mnist.npz`` bundle (the widely mirrored
Keras copy) and cached next to this file. Subsequent runs load from disk.

Run directly to pre-download without training::

    python -m data.mnist_loader
"""

import os
import sys
import urllib.request

import numpy as np

# Primary + fallback mirrors of the same ~11 MB npz bundle.
_URLS = [
    "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz",
    "https://s3.amazonaws.com/img-datasets/mnist.npz",
]
_CACHE_DIR = os.path.dirname(os.path.abspath(__file__))
_CACHE_PATH = os.path.join(_CACHE_DIR, "mnist.npz")


def _report(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100.0, downloaded * 100.0 / total_size)
        mb = downloaded / 1e6
        sys.stdout.write(f"\r  downloading mnist.npz: {pct:5.1f}%  ({mb:5.1f} MB)")
        sys.stdout.flush()


def download(force=False):
    """Ensure ``mnist.npz`` exists locally; return its path."""
    if os.path.exists(_CACHE_PATH) and not force:
        return _CACHE_PATH

    last_err = None
    for url in _URLS:
        try:
            print(f"Fetching MNIST from {url}")
            urllib.request.urlretrieve(url, _CACHE_PATH, _report)
            print("\n  done.")
            return _CACHE_PATH
        except Exception as exc:  # try the next mirror
            last_err = exc
            print(f"\n  failed ({exc}); trying next mirror...")

    raise RuntimeError(f"could not download MNIST from any mirror: {last_err}")


def _one_hot(labels, num_classes=10):
    out = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
    out[np.arange(labels.shape[0]), labels] = 1.0
    return out


def load_data(val_split=0.1, flatten=True, normalize=True, one_hot=True, seed=0):
    """Load MNIST and slice a validation set out of the training split.

    Returns ``(X_train, y_train), (X_val, y_val), (X_test, y_test)``.

    Parameters
    ----------
    val_split : float
        Fraction of the 60k training images held out for validation.
    flatten : bool
        Reshape 28x28 images to length-784 vectors.
    normalize : bool
        Scale pixel values from [0, 255] into [0, 1].
    one_hot : bool
        One-hot encode the integer labels into length-10 vectors.
    """
    path = download()
    with np.load(path) as f:
        x_train, y_train = f["x_train"], f["y_train"]
        x_test, y_test = f["x_test"], f["y_test"]

    def prep_x(x):
        x = x.astype(np.float32)
        if normalize:
            x /= 255.0
        if flatten:
            x = x.reshape(x.shape[0], -1)
        return x

    def prep_y(y):
        return _one_hot(y) if one_hot else y.astype(np.int64)

    x_train, x_test = prep_x(x_train), prep_x(x_test)
    y_train, y_test = prep_y(y_train), prep_y(y_test)

    # Carve out a validation set from a shuffled view of the training data.
    rng = np.random.default_rng(seed)
    perm = rng.permutation(x_train.shape[0])
    x_train, y_train = x_train[perm], y_train[perm]

    n_val = int(x_train.shape[0] * val_split)
    x_val, y_val = x_train[:n_val], y_train[:n_val]
    x_train, y_train = x_train[n_val:], y_train[n_val:]

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


if __name__ == "__main__":
    download()
    (xtr, ytr), (xva, yva), (xte, yte) = load_data()
    print(f"train: {xtr.shape} {ytr.shape}")
    print(f"val:   {xva.shape} {yva.shape}")
    print(f"test:  {xte.shape} {yte.shape}")

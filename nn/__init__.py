"""A small, modular neural-network framework built on NumPy.

Public API
----------
>>> from nn import Network, Dense, Activation, SGD, Adam, MSE, CrossEntropy
"""

from .layers import Layer, Dense, Activation, Dropout
from .conv import Conv2D, MaxPool2D, Flatten
from .losses import Loss, MSE, CrossEntropy
from .optimizers import Optimizer, SGD, Adam
from .network import Network

__all__ = [
    "Layer",
    "Dense",
    "Activation",
    "Dropout",
    "Conv2D",
    "MaxPool2D",
    "Flatten",
    "Loss",
    "MSE",
    "CrossEntropy",
    "Optimizer",
    "SGD",
    "Adam",
    "Network",
]

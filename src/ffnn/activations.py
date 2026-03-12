import numpy as np
from numpy.typing import NDArray
from typing import Literal

ActivationName = Literal["linear", "relu", "sigmoid", "tanh", "softmax"]


def linear(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return x


def relu(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.maximum(0.0, x)


def sigmoid(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return 1.0 / (1.0 + np.exp(-x))


def tanh(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.tanh(x)


def softmax(x: NDArray[np.float64]) -> NDArray[np.float64]:
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def apply_activation(
    name: ActivationName, x: NDArray[np.float64]
) -> NDArray[np.float64]:
    if name == "linear":
        return linear(x)
    if name == "relu":
        return relu(x)
    if name == "sigmoid":
        return sigmoid(x)
    if name == "tanh":
        return tanh(x)
    if name == "softmax":
        return softmax(x)
    raise ValueError(f"Unsupported activation: {name}")

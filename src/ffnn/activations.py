import numpy as np
from numpy.typing import NDArray
from typing import Literal

ActivationName = Literal["linear", "relu", "sigmoid", "tanh", "softmax"]


def linear(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return x

def linear_derivative(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.ones_like(x)

def relu(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.maximum(0.0, x)

def relu_derivative(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return (x > 0).astype(np.float64)

def sigmoid(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative(x: NDArray[np.float64]) -> NDArray[np.float64]:
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.tanh(x)

def tanh_derivative(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return 1.0 - np.tanh(x) ** 2

def softmax(x: NDArray[np.float64]) -> NDArray[np.float64]:
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def softmax_derivative(x: NDArray[np.float64]) -> NDArray[np.float64]:
    # bentuk awal dari softmax adalah N (jumlah sampel) x C (jumlah kelas)
    # hasilnya pengen bentuk matriks jacobian jadi perlu diubah jadi N x C x C
    s = softmax(x)
    
    # matriks diagonal kasusnya jika i == j, elemen diagonalnya s_i * (1 - s_i)
    diag_s = s[:, :, np.newaxis] * np.eye(s.shape[1])
    # s[:, :, np.newaxis] ngubah dari (N, C) jadi (N, C, 1)
    # np.eye(s.shape[1]) biki matriks identitas ukuran (C, C)
    # pas dikali bakal jadi (N, C, C) 
    
    # matriks outer kasusnya jika i != j, elemen outernya -s_i * s_j
    outer_s = s[:, np.newaxis, :] * s[:, :, np.newaxis]
    # s[:, np.newaxis, :] ngubah dari (N, C) jadi (N, 1, C)
    # s[:, :, np.newaxis] ngubah dari (N, C) jadi (N, C, 1)
    # pas dikali bakal jadi (N, C, C)
    
    jacobian = diag_s - outer_s
    return jacobian
        
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

def apply_activation_derivative(
    name: ActivationName, x: NDArray[np.float64]
) -> NDArray[np.float64]:
    if name == "linear":
        return linear_derivative(x)
    if name == "relu":
        return relu_derivative(x)
    if name == "sigmoid":
        return sigmoid_derivative(x)
    if name == "tanh":
        return tanh_derivative(x)
    if name == "softmax":
        return softmax_derivative(x)
    raise ValueError(f"Unsupported activation: {name}")
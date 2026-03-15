import numpy as np
from numpy.typing import NDArray
from typing import Literal

LossName = Literal["mse", "binary_crossentropy", "categorical_crossentropy"]

epsilon = 1e-15

def mse(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    loss = np.mean((y_true - y_pred) ** 2)
    return float(loss)

def mse_derivative(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> NDArray[np.float64]:
    n = y_true.shape[0]
    return (2 / n) * (y_pred - y_true)

def binary_crossentropy(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    y_pred_clip = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.mean(y_true * np.log(y_pred_clip) + (1 - y_true) * np.log(1 - y_pred_clip))
    return float(loss)

def binary_crossentropy_derivative(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> NDArray[np.float64]:
    y_pred_clip = np.clip(y_pred, epsilon, 1 - epsilon)
    n = y_true.shape[0]
    return - (1 / n) * (y_true / y_pred_clip - (1 - y_true) / (1 - y_pred_clip))

def categorical_crossentropy(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    y_pred_clip = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.mean(np.sum(y_true * np.log(y_pred_clip), axis=1))
    return float(loss)

def categorical_crossentropy_derivative(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> NDArray[np.float64]:
    y_pred_clip = np.clip(y_pred, epsilon, 1 - epsilon)
    n = len(y_true)
    return - (1 / n) * (y_true / y_pred_clip)

def apply_loss_function(name: LossName, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    if name == "mse":
        return mse(y_true, y_pred)
    if name == "binary_crossentropy":
        return binary_crossentropy(y_true, y_pred)
    if name == "categorical_crossentropy":
        return categorical_crossentropy(y_true, y_pred) 
    raise ValueError(f"Unsupported loss function: {name}")

def apply_loss_derivative(name: LossName, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> NDArray[np.float64]:
    if name == "mse":
        return mse_derivative(y_true, y_pred)
    if name == "binary_crossentropy":
        return binary_crossentropy_derivative(y_true, y_pred)
    if name == "categorical_crossentropy":
        return categorical_crossentropy_derivative(y_true, y_pred)
    raise ValueError(f"Unsupported loss function: {name}")

# regularisasi dipakai di hitung loss, turunannya dipakai di hitung gradien
def l1_regularization(weights: NDArray[np.float64], lambda_: float) -> float:
    return lambda_ * np.sum(np.abs(weights))

def l1_regularization_derivative(weights: NDArray[np.float64], lambda_: float) -> NDArray[np.float64]:
    return lambda_ * np.sign(weights)

def l2_regularization(weights: NDArray[np.float64], lambda_: float) -> float:
    return lambda_ * np.sum(weights ** 2)

def l2_regularization_derivative(weights: NDArray[np.float64], lambda_: float) -> NDArray[np.float64]:
    return 2 * lambda_ * weights
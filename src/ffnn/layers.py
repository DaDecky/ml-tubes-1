from typing import Literal, Optional

import numpy as np
from numpy.typing import NDArray

from .losses import apply_loss_derivative

from .activations import ActivationName, apply_activation, apply_activation_derivative

WeightInitializer = Literal["zeros", "random_uniform", "random_normal"]


class Dense:
    def __init__(
        self,
        n_neuron: int,
        activation: ActivationName,
        input_dim: Optional[int] = None,
        weight_initializer: WeightInitializer = "random_normal",
        seed: Optional[int] = None,
        lower_bound: float = -0.5,
        upper_bound: float = 0.5,
        mean: float = 0.0,
        variance: float = 1.0,
    ) -> None:
        self._n_neuron: int = n_neuron
        self._input_dim: Optional[int] = input_dim
        self._activation: ActivationName = activation
        self._weight_initializer: WeightInitializer = weight_initializer
        self._lower_bound: float = lower_bound
        self._upper_bound: float = upper_bound
        self._mean: float = mean
        self._variance: float = variance
        self._rng = np.random.default_rng(seed)

        self._is_first_layer = input_dim is not None
        self._weights: Optional[NDArray[np.float64]] = None
        self._bias: Optional[NDArray[np.float64]] = None
        self._last_input: Optional[NDArray[np.float64]] = None
        self._last_linear_output: Optional[NDArray[np.float64]] = None
        self._last_output: Optional[NDArray[np.float64]] = None

        self._error_terms: NDArray[np.float64] = None

        if input_dim is not None:
            self.build(input_dim)

    def is_first_layer(self) -> bool:
        return self._is_first_layer
    
    def last_output(self) -> NDArray[np.float64]:
        return self._last_output
    
    def last_input(self) -> NDArray[np.float64]:
        return self._last_input

    def build(self, input_dim: int) -> None:
        self._input_dim = input_dim
        self._weights = self._initialize_weights(input_dim)
        self._bias = self._initialize_bias()

    def forward(self, inputs: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._weights is None or self._bias is None:
            raise ValueError("Layer weights are not initialized. Call build() first.")

        batch_inputs = np.asarray(inputs, dtype=np.float64)
        if batch_inputs.ndim == 1:
            batch_inputs = batch_inputs.reshape(1, -1)

        if batch_inputs.shape[1] != self._input_dim:
            raise ValueError(
                f"Expected input with {self._input_dim} features, got {batch_inputs.shape[1]}"
            )

        self._last_input = batch_inputs
        self._last_linear_output = batch_inputs @ self._weights + self._bias
        self._last_output = apply_activation(self._activation, self._last_linear_output)
        return self._last_output
    
    def _compute_output_error_terms(self, loss, target, predicted, batch_size):
        loss_derivative = apply_loss_derivative(loss, target, predicted)
        activation_derivative = apply_activation_derivative(self._activation, self._last_linear_output)

        error_terms = loss_derivative * activation_derivative

        return error_terms

    def _compute_hidden_error_terms(self, prev_error_terms, prev_layer_weights, batch_size):

        prev_loss_derivative = prev_error_terms
        activation_derivative = apply_activation_derivative(self._activation, self._last_linear_output)

        error_terms = (prev_loss_derivative @ prev_layer_weights.T) * activation_derivative

        return error_terms
        

    def backward(self, lr: int, loss, target: Optional[NDArray[np.float64]]=None, predictions=[], prev_layer_weights: Optional[NDArray[np.float64]] = [], prev_error_terms: Optional[NDArray[np.float64]] = [], batch_size: int=1) -> NDArray[np.float64]:
        
        if target is not None: # output layer
            error_terms = self._compute_output_error_terms(loss, target, predictions, batch_size)
            
        else: # hidden layer
            error_terms = self._compute_hidden_error_terms(prev_error_terms, prev_layer_weights, batch_size)

        delta_weights = self._last_input.T @ error_terms
        self._weights -= lr*delta_weights

        delta_bias =  np.sum(error_terms, axis=0, keepdims=True)
        self._bias -= lr*delta_bias
                

        return error_terms

    def output_dim(self) -> int:
        return self._n_neuron

    @property
    def weights(self) -> NDArray[np.float64]:
        if self._weights is None:
            raise ValueError("Layer weights are not initialized.")
        return self._weights

    @property
    def bias(self) -> NDArray[np.float64]:
        if self._bias is None:
            raise ValueError("Layer bias is not initialized.")
        return self._bias

    def _initialize_weights(self, input_dim: int) -> NDArray[np.float64]:
        shape = (input_dim, self._n_neuron)
        return self._initialize_array(shape)

    def _initialize_bias(self) -> NDArray[np.float64]:
        return self._initialize_array((1, self._n_neuron))

    def _initialize_array(self, shape: tuple[int, ...]) -> NDArray[np.float64]:
        if self._weight_initializer == "zeros":
            return np.zeros(shape, dtype=np.float64)

        if self._weight_initializer == "random_uniform":
            return self._rng.uniform(
                self._lower_bound, self._upper_bound, size=shape
            ).astype(np.float64)

        if self._weight_initializer == "random_normal":
            std = np.sqrt(self._variance)
            return self._rng.normal(self._mean, std, size=shape).astype(np.float64)

        raise ValueError(f"Unsupported weight initializer: {self._weight_initializer}")

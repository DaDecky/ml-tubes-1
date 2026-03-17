import numpy as np
from numpy.typing import NDArray
from typing import Literal, Optional

GammaInitializer = Literal["zeros", "ones", "random_uniform", "random_normal"]


class RMSNormalization:
    def __init__(
        self,
        input_dim: Optional[int] = None,
        gamma_initializer: GammaInitializer = "ones",
        epsilon: float = 1e-8,
        seed: Optional[int] = None,
        lower_bound: float = -0.5,
        upper_bound: float = 0.5,
        mean: float = 0.0,
        variance: float = 1.0,
    ) -> None:
        self._input_dim: Optional[int] = input_dim
        self._gamma_initializer: GammaInitializer = gamma_initializer
        self._epsilon: float = epsilon
        self._lower_bound: float = lower_bound
        self._upper_bound: float = upper_bound
        self._mean: float = mean
        self._variance: float = variance
        self._rng = np.random.default_rng(seed)

        self._is_first_layer = input_dim is not None
        self._gammas: Optional[NDArray[np.float64]] = None
        self._last_input: Optional[NDArray[np.float64]] = None
        self._last_rms: Optional[NDArray[np.float64]] = None
        self._last_x_norm: Optional[NDArray[np.float64]] = None
        self._last_output: Optional[NDArray[np.float64]] = None
        self.dG: Optional[NDArray[np.float64]] = None

        if input_dim is not None:
            self.build(input_dim)

    def is_first_layer(self) -> bool:
        return self._is_first_layer

    def build(self, input_dim: int) -> None:
        self._input_dim = input_dim
        self._gammas = self._initialize_gammas(input_dim)

    def forward(self, inputs: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._gammas is None:
            raise ValueError("Layer weights are not initialized. Call build() first.")

        batch_inputs = np.asarray(inputs, dtype=np.float64)
        if batch_inputs.ndim == 1:
            batch_inputs = batch_inputs.reshape(1, -1)

        if batch_inputs.shape[1] != self._input_dim:
            raise ValueError(
                f"Expected input with {self._input_dim} features, got {batch_inputs.shape[1]}"
            )

        self._last_input = batch_inputs
        rms = np.sqrt(
            np.mean(batch_inputs * batch_inputs, axis=1, keepdims=True) + self._epsilon
        )
        x_norm = batch_inputs / rms
        self._last_rms = rms
        self._last_x_norm = x_norm
        self._last_output = x_norm * self._gammas
        return self._last_output

    def backward(
        self,
        loss,
        target: Optional[NDArray[np.float64]] = None,
        predictions: Optional[NDArray[np.float64]] = None,
        _gammas: Optional[NDArray[np.float64]] = None,
        prev_layer_weights: Optional[NDArray[np.float64]] = None,
        prev_error_terms: Optional[NDArray[np.float64]] = None,
        batch_size: int = 1,
    ) -> NDArray[np.float64]:
        if prev_error_terms is None:
            raise ValueError("prev_error_terms is required for RMSNormalization backward.")
        if self._last_input is None or self._last_rms is None or self._last_x_norm is None:
            raise ValueError("No cached forward pass found.")

        dy = prev_error_terms
        self.dG = np.sum(dy * self._last_x_norm, axis=0, keepdims=True)

        dx_norm = dy * self._gammas
        d = self._last_input.shape[1]
        mean_dx_x = np.mean(dx_norm * self._last_input, axis=1, keepdims=True)
        dx = dx_norm / self._last_rms - self._last_input * mean_dx_x / (
            (self._last_rms ** 3) * d
        )
        return dx

    def output_dim(self) -> int:
        if self._input_dim is None:
            raise ValueError("Layer is not built.")
        return self._input_dim

    @property
    def input_dim(self) -> int:
        if self._input_dim is None:
            raise ValueError("Layer is not built.")
        return self._input_dim

    @property
    def gammas(self) -> NDArray[np.float64]:
        if self._gammas is None:
            raise ValueError("Layer weights are not initialized.")
        return self._gammas

    def _initialize_gammas(self, input_dim: int) -> NDArray[np.float64]:
        shape = (1, input_dim)
        return self._initialize_array(shape)

    def _initialize_array(self, shape: tuple[int, ...]) -> NDArray[np.float64]:
        if self._gamma_initializer == "zeros":
            return np.zeros(shape, dtype=np.float64)

        if self._gamma_initializer == "ones":
            return np.ones(shape, dtype=np.float64)

        if self._gamma_initializer == "random_uniform":
            return self._rng.uniform(
                self._lower_bound, self._upper_bound, size=shape
            ).astype(np.float64)

        if self._gamma_initializer == "random_normal":
            std = np.sqrt(self._variance)
            return self._rng.normal(self._mean, std, size=shape).astype(np.float64)

        raise ValueError(f"Unsupported gamma initializer: {self._gamma_initializer}")

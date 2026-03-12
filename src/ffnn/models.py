from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .layers import Dense


class Sequential:
    def __init__(self) -> None:
        self._layers: list[Dense] = []

    def add(self, layer: Dense) -> None:
        # Throw errors if invalids layer is added
        if len(self._layers) == 0 and not layer.is_first_layer():
            raise ValueError("First layer must have input dimension specified")
        # Throw error if subsequent layer is first layer
        if len(self._layers) > 0 and layer.is_first_layer():
            raise ValueError("Subsequent layer must not have input dimension specified")

        if len(self._layers) > 0:
            previous_layer = self._layers[-1]
            layer.build(previous_layer.output_dim())

        self._layers.append(layer)

    def compile(
        self,
        loss: Literal["mse", "binary_crossentropy", "categorical_crossentropy"],
    ) -> None:
        self._loss = loss

    def forward(self, inputs: NDArray[np.float64]) -> NDArray[np.float64]:
        output = np.asarray(inputs, dtype=np.float64)
        for layer in self._layers:
            output = layer.forward(output)
        return output

    def predict(self, inputs: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.forward(inputs)

    @property
    def layers(self) -> list[Dense]:
        return self._layers

    def summary(self) -> None:
        for i, layer in enumerate(self._layers):
            print(
                f"Layer {i}: Dense(input_dim={layer.weights.shape[0]}, "
                f"n_neuron={layer.output_dim()})"
            )

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from .losses import apply_loss_function

from .layers import Dense


class Sequential:
    def __init__(self) -> None:
        self._layers: list[Dense] = []
        self._learning_rate: float = 0.1

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
        learning_rate: float = 0.1,
    ) -> None:
        self._loss = loss
        self._learning_rate = learning_rate

    def forward(self, inputs: NDArray[np.float64]) -> NDArray[np.float64]:
        output = np.asarray(inputs, dtype=np.float64)
        for layer in self._layers:
            output = layer.forward(output)
        return output
    
    

    # ngupdate delta dan bobot
    def backward(self, target: NDArray[np.float64], outputs: NDArray[np.float64], batch_size: int):
        lr = self._learning_rate
        predictions = np.asarray(outputs, dtype=np.float64)


        layer_size = len(self._layers)
        last_layer = self._layers[-1]
        error_terms = last_layer.backward(predictions=predictions,target=target, lr=lr, loss=self._loss, batch_size=batch_size)
        for i in range(layer_size-2, 0, -1):
            prev_layer_weights = 1
            error_terms = self._layers[i].backward(prev_error_terms=error_terms, prev_layer_weights=prev_layer_weights, lr=lr, loss=self._loss, batch_size=batch_size) # TODO: ganti learning rate jangan di sini




    def predict(self, inputs: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.forward(inputs)
    

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64], epochs: int=10, batch_size: int=1) -> None:
        y = y.reshape(-1,1) # kudu reshape biar bentuknya sama kayak predictions
        y_size = len(y)
        x_size = len(x)

        if x_size != y_size:
            raise ValueError(f"JRows Features and Target Different in Size!")
        
        for epoch in range(epochs):
            for i in range(0, x_size, batch_size):

                x_batch = x[i : i + batch_size]
                y_batch = y[i : i + batch_size]
                
                predictions = self.forward(x_batch)
                loss = apply_loss_function(self._loss, y_batch, predictions)
                self.backward(y_batch, predictions, batch_size)
            
    


    @property
    def layers(self) -> list[Dense]:
        return self._layers

    def summary(self) -> None:
        for i, layer in enumerate(self._layers):
            print(
                f"Layer {i}: Dense(input_dim={layer.weights.shape[0]}, "
                f"n_neuron={layer.output_dim()})"
            )

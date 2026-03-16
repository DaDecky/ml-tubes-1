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
        loss: Literal["mse", "binary_crossentropy", "categorical_crossentropy"]
    ) -> None:
        last_activation = self._layers[-1].activation

        if last_activation == "softmax" and loss != "categorical_crossentropy":
            raise SystemError("Warning: softmax biasanya dipakai dengan categorical_crossentropy")

        if last_activation == "sigmoid" and loss == "categorical_crossentropy":
            raise SystemError("Warning: sigmoid tidak cocok untuk multiclass classification")
        self._loss = loss

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
        for i in range(layer_size-2, -1, -1):
            prev_layer_weights = self._layers[i+1].weights
            error_terms = self._layers[i].backward(prev_error_terms=error_terms, prev_layer_weights=prev_layer_weights, lr=lr, loss=self._loss, batch_size=batch_size) # TODO: ganti learning rate jangan di sini

    def predict(self, inputs: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.forward(inputs)

    # fit perlu ada dataset train sama validation biar nampilin loss per epochnya
    def fit(
        self,
        x_train: NDArray[np.float64], 
        y_train: NDArray[np.float64],
        x_val: NDArray[np.float64]=None,
        y_val: NDArray[np.float64]=None, 
        epochs: int=10, 
        batch_size: int=1,
        learning_rate: float=0.1,
        verbose: int=0
    ) -> dict[str, list[float]]:
        self._learning_rate = learning_rate
        
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1) # kudu reshape biar bentuknya sama kayak predictions
        if y_val is not None and y_val.ndim == 1:
            y_val = y_val.reshape(-1, 1) # reshape juga buat validation
        
        y_size = len(y_train)
        x_size = len(x_train)
        
        if x_size != y_size:
            raise ValueError(f"Number of Features and Target Different in Size!")

        history = {
            "training_loss": [],
            "validation_loss": []
        }
        
        for epoch in range(epochs):
            epoch_training_loss = 0.0
            
            for i in range(0, x_size, batch_size):

                x_batch = x_train[i : i + batch_size]
                y_batch = y_train[i : i + batch_size]
                current_batch_size = len(x_batch)
                
                predictions = self.forward(x_batch)
                batch_loss = apply_loss_function(self._loss, y_batch, predictions)
                epoch_training_loss += batch_loss * current_batch_size
                
                self.backward(y_batch, predictions, current_batch_size)
            

            data_loss  = epoch_training_loss / x_size

            reg_loss = 0
            for layer in self._layers:
                if layer.kernel_regularizer is not None:
                    reg_loss += layer.kernel_regularizer(layer.weights)
            
            total_loss = data_loss + reg_loss
                
            history["training_loss"].append(total_loss)
            
            if x_val is not None and y_val is not None:
                val_predictions = self.forward(x_val)
                val_loss = apply_loss_function(self._loss, y_val, val_predictions)
                
                reg_loss = 0
                for layer in self._layers:
                    if layer.kernel_regularizer is not None:
                        reg_loss += layer.kernel_regularizer(layer.weights)

                val_loss += reg_loss
                history["validation_loss"].append(val_loss)
                
                if verbose == 1:
                    print(f"epoch {epoch} loss {total_loss} val_loss {val_loss}")
            else:
                if verbose == 1:
                    print(f"epoch {epoch} loss {total_loss}")
                    
        return history

    @property
    def layers(self) -> list[Dense]:
        return self._layers

    def summary(self) -> None:
        for i, layer in enumerate(self._layers):
            print(
                f"Layer {i}: Dense(input_dim={layer.weights.shape[0]}, "
                f"n_neuron={layer.output_dim()})"
            )

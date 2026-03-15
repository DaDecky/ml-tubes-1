import numpy as np

from ffnn.layers import Dense
from ffnn.models import Sequential


def main() -> None:
    model = Sequential()
    model.add(Dense(n_neuron=16, input_dim=4, activation="relu", seed=42))
    model.add(Dense(n_neuron=8, activation="sigmoid", seed=42))
    model.add(Dense(n_neuron=1, activation="sigmoid", seed=42))
    model.compile(loss="binary_crossentropy")

    x_batch = np.array(
        [
            [3.5, 0.0, 1.0, 7.2],
            [2.8, 1.0, 0.0, 6.4],
            [3.9, 2.0, 1.0, 8.1],
            [3.0, 4.0, 2.0, 8.0]
        ],
        dtype=np.float64,
    )

    y = np.array([1,2,3,4])

    model.fit(x_batch,y,epochs=3)

    predictions = model.forward(x_batch)

    print("Model summary:")
    model.summary()
    print()
    print("Input batch shape:", x_batch.shape)
    print("Prediction shape:", predictions.shape)
    print("Predictions:")
    print(predictions)


if __name__ == "__main__":
    main()

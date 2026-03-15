import numpy as np

from ffnn.layers import Dense
from ffnn.models import Sequential


def main() -> None:
    model = Sequential()
    model.add(Dense(n_neuron=16, input_dim=5, activation="relu", seed=42))
    model.add(Dense(n_neuron=8, activation="sigmoid", seed=42))
    model.add(Dense(n_neuron=1, activation="sigmoid", seed=42))
    model.compile(loss="binary_crossentropy")

    x_batch = np.array(
        [
            [1,1,1,1,1],
            [2,2,2,2,2],
            [3,3,3,3,3],
            [4,4,4,4,4]
        ],
        dtype=np.float64,
    )

    y = np.array([1,2,3,4])


    model.fit(x_batch, y, epochs=3, batch_size=len(x_batch))

    x_test = np.array(
        [
            [1,1,1,1,1],
            [2,2,2,2,2],
            [3,3,3,3,3],
            [4,4,4,4,4]
        ],
        dtype=np.float64,
    )

    predictions = model.forward(x_test)

    print("Model summary:")
    model.summary()
    print()
    print("Input batch shape:", x_batch.shape)
    print("Prediction shape:", predictions.shape)
    print("Predictions:")
    print(predictions)


if __name__ == "__main__":
    main()

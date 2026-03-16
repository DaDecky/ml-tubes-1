import numpy as np
import sys

from ffnn.optimizers import Adam
sys.path.append("src")

from ffnn.layers import Dense
from ffnn.models import Sequential


def main() -> None:
    model = Sequential()
    model.add(Dense(n_neuron=5, input_dim=5, activation="relu", seed=42))
    model.add(Dense(n_neuron=1, activation="linear", seed=42))
    model.compile(loss="mse", optimizer=Adam(0.001))

    # data train
    x_train = np.array(
        [
            [1,1,1,1,1],
            [2,2,2,2,2],
            [3,3,3,3,3],
            [4,4,4,4,4]
        ],
        dtype=np.float64,
    )

    y_train = np.array([1,2,3,4])

    # data validation
    x_val = np.array(
        [
            [5,5,5,5,5],
            [6,6,6,6,6]
        ],
        dtype=np.float64,
    )

    y_val = np.array([5,6])

    history = model.fit(
        x_train=x_train, 
        y_train=y_train, 
        x_val=x_val,
        y_val=y_val,
        epochs=500, 
        batch_size=2, 
        learning_rate=0.01,
        verbose=1
    )

    model.save("test_save_model.pkl") 
    
    load_model = Sequential.load("test_save_model.pkl")
    
    x_test = np.array(
        [
            [1,1,1,1,1],
            [2,2,2,2,2],
            [3,3,3,3,3],
            [4,4,4,4,4]
        ],
        dtype=np.float64,
    )

    predictions = load_model.forward(x_test)

    print("Model summary:")
    load_model.summary()
    print()
    print("Input batch shape:", x_test.shape)
    print("Prediction shape:", predictions.shape)
    print("Predictions:")
    print(predictions)


if __name__ == "__main__":
    main()

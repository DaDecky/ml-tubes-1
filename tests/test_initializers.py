import numpy as np

from ffnn.layers import Dense


def xavier() -> None:
    fan_in = 32
    fan_out = 16
    layer = Dense(
        n_neuron=fan_out,
        input_dim=fan_in,
        activation="relu",
        weight_initializer="xavier",
        seed=42,
    )

    limit = np.sqrt(6.0 / (fan_in + fan_out))

    print("Xavier Initialization:")
    print("weights shape:", layer.weights.shape)
    print("bias shape:", layer.bias.shape)
    print("expected limit:", limit)
    print("weights min/max:", float(layer.weights.min()), float(layer.weights.max()))
    print("bias min/max:", float(layer.bias.min()), float(layer.bias.max()))
    print()


def he() -> None:
    fan_in = 32
    fan_out = 16
    layer = Dense(
        n_neuron=fan_out,
        input_dim=fan_in,
        activation="relu",
        weight_initializer="he",
        seed=42,
    )

    expected_std = np.sqrt(2.0 / fan_in)

    print("He Initialization:")
    print("weights shape:", layer.weights.shape)
    print("bias shape:", layer.bias.shape)
    print("expected std:", expected_std)
    print("weights std:", float(layer.weights.std()))
    print("bias std:", float(layer.bias.std()))
    print()


if __name__ == "__main__":
    xavier()
    he()

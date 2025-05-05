import numpy as np
from typing import List
from ml.layer import Layer


class Model:
    """
    A class that represents a feedforward neural network model. It is made up of one or more layers.
    """

    def __init__(self, layers: List["Layer"]) -> None:
        """
        Initializes the model.
        :param layers: A list of Layer objects.
        """
        self.layers = layers
        self._connect_layers()

    def __repr__(self) -> str:
        """
        Returns a string representation of the model.
        """
        repr_str = "Model(\n"
        for i, layer in enumerate(self.layers):
            repr_str += f"  Layer {i}: {layer.units} neurons, activation={layer.activation_name}\n"
        repr_str += ")"
        return repr_str

    def _connect_layers(self) -> None:
        """
        Ensures all layers are properly connected (input sizes initialized).
        """
        for i in range(1, len(self.layers)):
            if self.layers[i].input_size is None:
                self.layers[i].set_input_size_from_previous_layer(self.layers[i - 1].units)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, learning_rate, threshold, epochs:int):
        """
        Trains the model.
        :param x_train: Training data.
        :param y_train: Training labels.
        :param learning_rate: Learning rate.
        :param threshold:
        :param epochs: Number of training epochs.
        """
        for epoch in range(epochs):
            for idx,iteration in enumerate(x_train):
                iteration = np.array(iteration)
                y_pred = self.forward(iteration)
                error = y_train[idx] - y_pred
                self.retropropagation(error, learning_rate)
            # Calculate MSE with new weights.
            errors = np.array([])
            predictions = np.array([])
            for iteration in x_train:
                y_pred = self.forward(iteration)
                error = y_train - y_pred
                errors = np.append(errors, error)
                predictions = np.append(predictions, y_pred)
            mse = np.mean(0.5 * errors ** 2)
            # Training exit condition
            if mse < threshold or (predictions.all() == y_train.all()):
                print(f"Training complete after {epoch + 1} epochs.")
                break
            if epoch % 20 == 0:
                print(f"Epoch :{epoch + 1}, MSE: {mse:.4f}")
        print(f"Training stopped after {epochs} epochs.")

    def forward(self, input_vector: np.ndarray) -> np.ndarray:
        """
        Performs a forward pass through the entire model.
        :param input_vector: Input data to the first layer.
        :return: Final output of the network.
        """
        output = input_vector
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def retropropagation(self, error, learning_rate):
        """
        Performs a backward pass through the entire model.
        :param error: Calculated error.
        :param learning_rate: Learning rate.
        """
        for layer in reversed(self.layers):
            error = layer.retropropagation(error, learning_rate)

    def save(self, path: str) -> None:
        """
        Saves the model to a file.
        :param path: Path to save the model to.
        """
        pass

    def load(self, path: str) -> None:
        """
        Loads the model from a file.
        :param path: Path to load the model from.
        """
        pass

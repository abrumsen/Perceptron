import numpy as np
from typing import List
from ml.layer import Layer
from ml.history import History
from ml.optimizer import Optimizer


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

    def compile(self, learning_algorithm:Optimizer):
        """
        Defines the model's training algorithm, learning rate & metrics.
        """
        pass

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, epochs:int, verbose=0) -> History:
        """
        Trains the model.
        """
        pass

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

    def backward(self):
        """
        Performs a backward pass through the entire model.
        """
        pass

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

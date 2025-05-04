import numpy as np
from ml.activation import Activation

class Neuron:
    """
    A class representing a single neuron in a neural network layer.
    """
    def __init__(self, input_size: int, activation: Activation) -> None:
        """
        Initializes a new neuron.
        :param input_size: The size of the input to the neuron.
        :param activation: The activation function of the neuron.
        """
        self.weights = np.random.normal(0,1,input_size)
        self.bias = np.random.randn()
        self.activation = activation
        self.a = None
        self.z = None
        self.inputs = None

    def __repr__(self) -> str:
        return f"Neuron(weights={self.weights})"

    def forward(self, inputs: np.ndarray) -> float:
        """
        Perform the forward pass through the neuron.
        :param inputs: The input vector or scalar.
        :return: The output of the neuron after activation.
        """
        self.inputs = inputs
        self.z = np.dot(inputs, self.weights) + self.bias
        # Apply activation function
        self.a = self.activation(self.z)
        return self.a
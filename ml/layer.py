import numpy as np
from ml.neuron import Neuron
from ml.activation import Activation


class Layer:
    """
    A class representing a single layer in a neural network. It is made up of one or more neurons.
    """

    def __init__(self, units:int, activation:str, input_size:int | None=None, **activation_kwargs) -> None:
        """
        Initializes a new neuron layer.
        :param units: The number of neurons in the layer.
        :param activation: The activation function of the layer.
        :param input_size: The size of the input to the neuron.
        :param activation_kwargs: Keyword arguments passed to the activation function.
        """
        self.units = units
        self.activation_name = activation
        self.input_size = input_size  # Only required for first layer
        self.neurons = []
        self.output = None
        self.inputs = None

        # Create activation object with kwargs
        self.activation = Activation(activation, **activation_kwargs)

        if input_size:
            self.initialize_neurons(input_size)

    def __repr__(self) -> str:
        """
        Returns a string representation of the layer.
        """
        return f"Layer(Neurons:{[neuron for neuron in self.neurons]}, Activation:{self.activation})"

    def initialize_neurons(self, input_dim:int) -> None:
        """
        Initializes the neurons of the layer.
        :param input_dim: The size of the input to the neuron.
        """
        self.neurons = [Neuron(input_dim, self.activation) for _ in range(self.units)]

    def forward(self, input_vector:np.ndarray) -> np.ndarray:
        """
        Performs the forward pass of the layer.
        :param input_vector: The input to the neuron(s).
        :return: The output of the neuron(s).
        """
        self.inputs = input_vector
        self.output = np.array([neuron.forward(input_vector) for neuron in self.neurons])
        return self.output

    def retropropagation(self, error_signal, learning_rate):
        """
        Performs the backward pass of the layer.
        :param error_signal: The error signal of the neuron(s).
        :param learning_rate: The learning rate.
        :return: accumulated error signal for previous layer.
        """
        prev_error = np.zeros_like(self.inputs)

        for i, neuron in enumerate(self.neurons):
            # Each neuron receives its own error signal
            delta = neuron.retropropagation(error_signal[i], learning_rate)
            # Accumulate error signals for previous layer
            prev_error += delta

        return prev_error

    def set_input_size_from_previous_layer(self, prev_layer_units:int) -> None:
        """
        Sets the size of the input to the number of neurons in the previous layer.
        Only used if no input_size is specified.
        :param prev_layer_units: The previous layer units.
        """
        if not self.neurons:
            self.initialize_neurons(prev_layer_units)

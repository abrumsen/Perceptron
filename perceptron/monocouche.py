from enum import Enum
import pandas as pd
import numpy as np

from perceptron import PerceptronSimple, PerceptronGradient, PerceptronAdaline


class NeuronType(Enum):
    BASIC = "Basic"
    ADALINE = "Adaline"
    GRADIENT = "Gradient"


class Layer:
    def __init__(self, nbr_neurones: int, input_size: int, neurone_type: NeuronType, learning_rate: float = 0.3,
                 epochs: int = 1000):
        if not isinstance(neurone_type, NeuronType):
            raise ValueError("neurone_type must be an instance of NeuronType Enum")

        self.nbr_neurones = nbr_neurones
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.neurone_type = neurone_type
        self.neurons = self._create_neurons()
        self.histories = []

    def _create_neurons(self):
        neurons = []
        for _ in range(self.nbr_neurones):
            neuron = self._instantiate_neuron()
            neurons.append(neuron)
        return neurons

    def _instantiate_neuron(self):
        if self.neurone_type == NeuronType.BASIC:
            return PerceptronSimple(self.input_size, self.learning_rate, self.epochs)
        elif self.neurone_type == NeuronType.ADALINE:
            return PerceptronAdaline(self.input_size, self.learning_rate, self.epochs)
        elif self.neurone_type == NeuronType.GRADIENT:
            return PerceptronGradient(self.input_size, self.learning_rate, self.epochs)
        else:
            raise ValueError(f"Unsupported neuron type: {self.neurone_type}")

    def predict(self, inputs : np.array):
        raw_predictions = np.array([neuron.predict(inputs) for neuron in self.neurons])
        rounded_predictions = np.array([neuron.round_predict(inputs) for neuron in self.neurons])

        return raw_predictions.T, rounded_predictions.T

    def train_layer(self, dataset:pd.DataFrame, seuil:float = 0, initiate_weights_zero:bool = True):
        self.histories = []

        for idx, neurone in enumerate(self.neurons):
            custom_data = pd.DataFrame({
                "inputs": dataset["inputs"],
                "label": dataset["label"].apply(lambda labels: labels[idx])
            })

            input_shape = len(dataset["inputs"].iloc[0])
            if initiate_weights_zero:
                neurone.weights = np.zeros(input_shape, dtype=float)

            if isinstance(neurone, PerceptronSimple):
                history = neurone.train(custom_data)
            elif isinstance(neurone, PerceptronGradient):
                history = neurone.mode_choose(custom_data, seuil=seuil, mode="classification")
            elif isinstance(neurone, PerceptronAdaline):
                history = neurone.train_classification(custom_data, seuil=seuil, until_no_error=True)
            else:
                raise ValueError("Unknown neuron type")

            self.histories.append(history)

    def get_history(self, neurone_index=None):
        if neurone_index is None:
            return self.histories
        if 0 <= neurone_index < len(self.histories):
            return self.histories[neurone_index]
        raise IndexError("Neuron index out of range")

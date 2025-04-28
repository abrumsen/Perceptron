from enum import Enum
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

    def predict(self, inputs):

        return [neuron.predict(inputs) for neuron in self.neurons]

    def train_layer(self,data, seuil=0.01, mode_classification=True):

        for idx, neurone in enumerate(self.neurons):
            print(f"\n--- Training in progress {idx + 1} ---")
            if isinstance(neurone, PerceptronSimple):
                neurone.train(data)
            elif isinstance(neurone, PerceptronGradient):
                neurone.mode_choose(data, seuil=seuil, mode=mode_classification)
            elif isinstance(neurone, PerceptronAdaline):
                neurone.train(data, seuil=seuil)
            else:
                raise ValueError("Unknown neuron type")
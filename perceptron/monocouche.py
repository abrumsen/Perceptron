from enum import Enum
import numpy as np
import pandas as pd

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
        predict = [neuron.predict(inputs) for neuron in self.neurons]
        round_predict = [neuron.round_predict(inputs) for neuron in self.neurons]

        return predict, round_predict

    def train_layer(self, data, seuil=0.01, mode_classification=True):

        for idx, neurone in enumerate(self.neurons):
            print(f"\n--- Training in progress for neuron n°{idx + 1} ---")
            custom_data = pd.DataFrame({
                "inputs": data["inputs"],
                "label": data["labels"].apply(lambda labels: labels[idx])
            })
            if isinstance(neurone, PerceptronSimple):
                # neurone.weights = np.array(
                #     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                #      0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                neurone.train(custom_data)
            elif isinstance(neurone, PerceptronGradient):
                # neurone.weights = np.array(
                #     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                #      0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                neurone.mode_choose(custom_data, seuil=seuil, mode=mode_classification)
            elif isinstance(neurone, PerceptronAdaline):
                # neurone.weights = np.array(
                #     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                #      0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                neurone.train(custom_data, seuil=seuil)
            else:
                raise ValueError("Unknown neuron type")


# data = pd.DataFrame({
#     "inputs": [np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
#                np.array([1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]),
#                np.array([1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1]),
#                np.array([1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0])],
#     "labels": [np.array([1, -1, -1, -1]), np.array([-1, 1, -1, -1]), np.array([-1, -1, 1, -1]),
#                np.array([-1, -1, -1, 1])]
# })
# predict_data_void = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
# predict_data_cross = np.array([1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0])
# macouche = Layer(nbr_neurones=4, input_size=25, neurone_type=NeuronType.ADALINE, learning_rate=0.001, epochs=1000)
# macouche.train_layer(data=data, seuil=0)
# print(f"\nResults :\n{macouche.predict(predict_data_void)}")

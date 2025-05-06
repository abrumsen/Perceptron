import numpy as np
import pandas as pd

from perceptron import Perceptron
from utils.history import History


class PerceptronSimple(Perceptron):
    def __init__(self, input_size: int, learning_rate: float = 0.3, epochs: int = 1000) -> None:
        super().__init__(input_size, learning_rate, epochs)

    @staticmethod
    def activation_function(potential: float) -> int:
        """
        Defines the activation function of the perceptron.
        :param potential: The calculated potential.
        :return: 1 if activated, 0 if not.
        """
        return 1 if potential >= 0 else 0

    def predict(self, inputs: np.ndarray) -> int:
        """
        Calculates the prediction of the perceptron.
        :param inputs: The input vector (includes bias).
        :return: 1 if activated, 0 if not.
        """
        return self.activation_function(np.dot(self.weights, inputs))

    def correct(self, error: float, x: np.ndarray) -> None:
        """
        Correct synaptic weights using the basic perceptron rule.
        :param error: The difference between the predicted and expected value.
        :param x: The input vector (includes bias).
        """
        self.weights += self.learning_rate * error * x

    def train(self, training_data: pd.DataFrame) -> History:
        """
        Trains the perceptron with the given training data using the basic perceptron algorithm.
        :param training_data: The training data dataframe.
        :return: 1 if training was successful, 0 if not.
        """
        history = History()
        for epoch in range(self.epochs):
            errors = 0
            for _, row in training_data.iterrows():
                prediction = self.predict(row["inputs"])
                error = row["label"] - prediction
                if error != 0:
                    errors += 1
                    self.correct(error, row["inputs"])
            if errors == 0:
                print(f"Training complete for {epoch + 1} epochs")
                return history
        print(f"Training stopped after {self.epochs} epochs")
        return history

    def export_neuron(self, path: str) -> None:
        """
        Export the perceptron weights to a csv file.
        :param path: The path to save the weights to.
        """
        columns = [f"w{i}" for i in range(len(self.weights))]
        df = pd.DataFrame(self.weights, columns=columns)
        df.to_csv(path, index=False)

    def import_neuron(self, path: str) -> None:
        """
        Import the perceptron weights.
        :param path: The path to load the weights from.
        """
        pass
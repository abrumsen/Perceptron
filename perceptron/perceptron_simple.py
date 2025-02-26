import numpy as np
import pandas as pd

from perceptron.perceptron_base import Perceptron

class PerceptronSimple(Perceptron):
    def __init__(self, input_size: int, learning_rate: float=0.3, epochs: int=1000) -> None:
        super().__init__(input_size, learning_rate, epochs)

    def activation_function(self, potential: float) -> int:
        return 1 if potential > 0 else 0

    def predict(self, inputs: np.ndarray) -> int:
        return self.activation_function(np.dot(self.weights, inputs))

    def correct(self, error: float, x: np.ndarray) -> None:
        for i in range(len(x)):
            self.weights[i] += self.learning_rate * error * x[i]

    def train(self, training_data: pd.DataFrame) -> None:
        for epoch in range(self.epochs):
            errors = 0
            for index, row in training_data.iterrows():
                prediction = self.predict(row["inputs"])
                error = row["label"] - prediction
                if error != 0:
                    errors += 1
                    self.correct(error, row["inputs"])
            if not errors:
                print(f"Training complete for {epoch} epochs")
                break
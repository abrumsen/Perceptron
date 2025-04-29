import numpy as np
import pandas as pd
from numpy import ndarray

from perceptron import Perceptron

class PerceptronAdaline(Perceptron):
    def __init__(self, input_size:int, learning_rate: float=0.3, epochs: int=1000)-> None:
        super().__init__(input_size, learning_rate, epochs)
        self.losses = []

    @staticmethod
    def activation_function(array_y: np.ndarray) -> np.ndarray:
        array_s = np.where(array_y >= 0, 1, -1)
        return array_s

    def predict(self, data: np.ndarray) -> ndarray:
        # return self.activation_function(np.dot(self.weights, data))
        return np.dot(self.weights, data)

    def round_predict(self, data: np.ndarray) -> ndarray:
        # return self.activation_function(np.dot(self.weights, data))
        array = np.dot(self.weights, data)
        return self.activation_function(array)

    def correct_weights(self, expected_value, actual_value, inputs):
        self.weights += self.learning_rate * (expected_value - actual_value) * inputs
        return self.weights

    @staticmethod
    def quadratic_error(expected_value:np.array, actual_value:np.array):
        error = (expected_value - actual_value)**2
        error_sum = 0
        for e in error:
            error_sum += e
        return 0.5*error_sum

    def mean_quadratic_error(self, expected_value:np.array, inputs:np.ndarray) -> float:
        output_recalculation = np.array([])
        for inpt in inputs:
            prediction = self.predict(inpt)
            output_recalculation = np.append(output_recalculation, prediction)

        quadratic_error = self.quadratic_error(expected_value, output_recalculation)
        return quadratic_error/len(output_recalculation)

    def train(self, dataset: pd.DataFrame, seuil: float) -> bool:
        training_data = dataset["inputs"].values
        expected_values = dataset["label"].values
        for epoch in range(self.epochs):

            obtained_values = np.array([])
            for i in range(len(training_data)):
                data = training_data[i]
                expected_v = expected_values[i]
                prediction = self.predict(data)
                error = expected_v - prediction
                self.correct_weights(expected_v, prediction, data)
                obtained_values = np.append(obtained_values,prediction)
            mean_quad_error = self.mean_quadratic_error(expected_values, training_data)
            if mean_quad_error < seuil:
                print(
                    f"Training complete for {epoch + 1} epochs, epochs with quad error {mean_quad_error} and error {error}\nw0 : {self.weights[0]}\nw1 : {self.weights[1]}\nw2 : {self.weights[2]}\nThe obtained values: {self.activation_function(obtained_values)}, the expected values: {expected_values}")
                # print(f"Training complete for {epoch + 1} epochs, epochs with quad error {mean_quad_error} and error {error}\nw0 : {self.weights[0]}\nw1 : {self.weights[1]}\nw2 : {self.weights[2]}\nThe obtained values: {obtained_values}, the expected values: {expected_values}")
                return True
        # print(f"Training uncomplete after {self.epochs} epochs, epochs with error {mean_quad_error} and error {error}\nw0 : {self.weights[0]}\nw1 : {self.weights[1]}\nw2 : {self.weights[2]}\nThe obtained values: {obtained_values}, the expected values: {expected_values}")
        print(f"Training uncomplete after {self.epochs} epochs, epochs with quad error {mean_quad_error} and error {error}\nw0 : {self.weights[0]}\nw1 : {self.weights[1]}\nw2 : {self.weights[2]}\nThe obtained values: {self.activation_function(obtained_values)}, the expected values: {expected_values}")
        return False





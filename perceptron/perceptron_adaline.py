import numpy as np
import pandas as pd

from perceptron import Perceptron

class PerceptronAdaline(Perceptron):
    def __init__(self, input_size:int, learning_rate: float=0.3, epochs: int=1000)-> None:
        super().__init__(input_size, learning_rate, epochs)
        self.losses = []

    def activation_function(array_y: np.ndarray) -> np.ndarray:
        return np.where(array_y >= 0, 1, -1)

    def predict(self, data: np.ndarray) -> np.ndarray:
        return self.activation_function(np.dot(self.weights, data))

    def correct_weights(self, expected_value, actual_value, inputs):
        self.weights += self.learning_rate * (actual_value - expected_value) * inputs
        return self.weights

    def quadratic_error(self, expected_value:np.array, actual_value:np.array):
        error = 0.5*(actual_value - expected_value)**2
        error_sum = 0
        for e in error:
            error_sum += e
        return error_sum

    def mean_quadratic_error(self, expected_value:np.array, actual_value:np.array):
        quadratic_error = self.quadratic_error(expected_value, actual_value)
        return quadratic_error/len(actual_value)




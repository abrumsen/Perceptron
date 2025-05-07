import numpy as np
import pandas as pd
from numpy import ndarray

from perceptron import Perceptron
from utils.history import History


class PerceptronAdaline(Perceptron):
    def __init__(self, input_size:int, learning_rate: float=0.3, epochs: int=1000)-> None:
        super().__init__(input_size, learning_rate, epochs)
        self.losses = []

    @staticmethod
    def activation_function(array_y: np.ndarray) -> np.ndarray:
        array_s = np.where(array_y >= 0, 1, -1)
        return array_s

    def predict(self, data: np.ndarray) -> ndarray:
        return np.dot(data,self.weights)

    def round_predict(self, data: np.ndarray) -> ndarray:
        array = np.dot(self.weights, data)
        return self.activation_function(array)

    def correct_weights(self, expected_value, actual_value, inputs):
        self.weights += self.learning_rate * (expected_value - actual_value) * inputs
        return self.weights

    @staticmethod
    def quadratic_error(expected_value:np.array, actual_value:np.array):
        return 0.5*np.sum((expected_value - actual_value)**2)

    def mean_quadratic_error(self, expected_value:np.array, inputs:np.ndarray) -> float:
        return self.quadratic_error(expected_value, inputs)/len(inputs)

    def train_classification(self, dataset: pd.DataFrame, seuil: float, until_no_error: bool=False) -> History:
        training_data = np.stack(dataset["inputs"].values)
        expected_values = dataset["label"].values
        history = History()
        for epoch in range(self.epochs):
            predictions_epoch = np.zeros_like(expected_values, dtype=float)

            for i in range(len(training_data)):
                x_i = training_data[i]
                y_i = expected_values[i]

                prediction = self.predict(x_i)
                error = y_i - prediction
                self.weights += self.learning_rate * error * x_i
                predictions_epoch[i] = prediction
            # actual_prediction = self.predict(training_data)
            # mean_quad_error = self.mean_quadratic_error(expected_values, actual_prediction)
            # accuracy = np.mean(self.activation_function(actual_prediction) == expected_values)
            mean_quad_error = self.mean_quadratic_error(expected_values, predictions_epoch)
            accuracy = np.mean(self.activation_function(predictions_epoch) == expected_values)
            history.log(epoch=epoch+1, mse=mean_quad_error, accuracy=accuracy, weights=self.weights.copy())

            if until_no_error :
                if mean_quad_error < seuil or accuracy == 1.0:
                    return history
            else:
                if mean_quad_error < seuil :
                    return history
        return history

    def train_regression(self, dataset: pd.DataFrame, seuil: float) -> History:
        """
        Trains the perceptron with regression using the given training data.
        Stops when the error is below the specified threshold or after completing all epochs.
        :param dataset: A DataFrame containing the training data with inputs and labels.
        :param seuil: The threshold for error to stop training.
        :return: history if training is successful, history if it stops after all epochs.
        """
        training_data = np.stack(dataset["inputs"].values)
        expected_values = dataset["label"].values
        history = History()
        for epoch in range(self.epochs):
            predictions_epoch = np.zeros_like(expected_values, dtype=float)
            for i in range(len(training_data)):
                x_i = training_data[i]
                y_i = expected_values[i]
                prediction = self.predict(x_i)
                error = y_i - prediction
                self.weights += self.learning_rate * error * x_i
                predictions_epoch[i] = prediction
            actual_prediction = self.predict(training_data)
            mean_quad_error = self.mean_quadratic_error(expected_values, actual_prediction)
            history.log(epoch=epoch + 1, mse=mean_quad_error, weights=self.weights.copy())
            if epoch % 1000 == 0 :
                print(mean_quad_error)
            if mean_quad_error < seuil:
                return history
        return history



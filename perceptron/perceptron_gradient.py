import numpy as np
import pandas as pd

from perceptron import Perceptron
from utils.history import History

class PerceptronGradient(Perceptron):
    def __init__(self, input_size: int, learning_rate: float = 0.3, epochs: int = 1000) -> None:
        super().__init__(input_size, learning_rate, epochs)


    @staticmethod
    def activation_function(array_y: np.ndarray) -> np.ndarray:
        """
        Defines the activation function of the perceptron.
        Converts the y array into an array of either 1 or -1.
        :param array_y: The array where the y where calculated.
        :return: An array containing 1 if y >= 0, otherwise -1.
        """
        array_s = np.where(array_y >= 0, 1, -1)
        return array_s


    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Calculate the y for the given data using the current weights.
        :param data: A DataFrame containing input data (x).
        :return: An array containing the y values of the perceptron.
        """
        # Conversion de la colonne "inputs" en une matrice NumPy
        input_matrix = np.vstack(data["inputs"].values)  # Empile les listes en une matrice NumPy
        # Calcul des y
        array_y = np.dot(input_matrix, self.weights)  # Equivalant à w0*x0 + w1*x1 + wn*xn
        return array_y


    @staticmethod
    def error(array_y:np.ndarray, array_d: np.ndarray) -> np.ndarray:
        """
        Computes the error for a single epoch of training in mode regression.
        :param array_y: The predicted outputs from the perceptron.
        :param array_d The actual outputs.
        :return: The error value for each sample in the dataset.
        """
        array_error = 0.5*(array_y - array_d)**2
        return array_error


    def correct(self, array_y: np.ndarray, array_d: np.ndarray, array_x: np.ndarray) -> None:
        """
        Updates the perceptron weights based on the error correction.
        :param array_y: The predicted outputs from the perceptron.
        :param array_d: The desired outputs (target values).
        :param array_x: The input features.
        """
        array_error_to_correct = array_d - array_y
        self.weights += self.learning_rate * np.dot(array_error_to_correct, array_x)


    def classification_train(self, training_data: pd.DataFrame, seuil: float, until_no_error:bool) -> History:
        """
        Trains the perceptron with classification using the given training data.
        Stops when the error is below the specified threshold or after completing all epochs.
        :param training_data: A DataFrame containing the training data with inputs and labels.
        :param seuil: The threshold for error to stop training.
        :return: history if training is successful, history if it stops after all epochs.
        """
        history = History()
        for epoch in range(self.epochs):
            y = self.predict(training_data)
            s = self.activation_function(y)
            d = training_data["label"].values
            error = self.error(y, d)
            history.log(epoch=epoch, mse=np.mean(error), accuracy=np.mean(s == training_data["label"].values))
            if np.mean(error) <= seuil and (s == training_data["label"].values).all() and until_no_error == True:
                print(f"Training complete after {epoch + 1} epochs.")
                return history
            elif np.mean(error) <= seuil or (s == training_data["label"].values).all() and until_no_error == False:
                print(f"Training complete after {epoch + 1} epochs.")
                return history
            self.correct(y, training_data["label"], training_data["inputs"])
        print(f"Training stopped after {epoch + 1} epochs with error {np.mean(error)}")
        return history


    def regression_train(self, training_data: pd.DataFrame, seuil: float) -> int:
        """
        Trains the perceptron with regression using the given training data.
        Stops when the error is below the specified threshold or after completing all epochs.
        :param training_data: A DataFrame containing the training data with inputs and labels.
        :param seuil: The threshold for error to stop training.
        :return: history if training is successful, history if it stops after all epochs.
        """
        history = History()
        for epoch in range(self.epochs):
            y = self.predict(training_data)
            d = training_data["label"].values
            error = self.error(y,d)
            history.log(epoch=epoch, mse=np.mean(error))
            if np.mean(error) <= seuil:
                print(f"Training complete after {epoch + 1} epochs.")
                return history
            self.correct(y, training_data["label"], training_data["inputs"])
        print(f"Training stopped after {epoch + 1} epochs with error {np.mean(error)}")
        return history


    def mode_choose(self, training_data: pd.DataFrame, seuil: float, mode: str, until_no_error:bool) -> int or str:
        """
        Chooses the training mode for the perceptron: classification or regression.
        Executes the corresponding training method based on the specified mode.
        :param training_data: A DataFrame containing the training data with inputs and labels.
        :param seuil: The threshold for error to stop training.
        :param mode: A string indicating the training mode.
        :return: history if training is successful, history if it stops after all epochs.
        """
        if mode == "classification":
            return self.classification_train(training_data, seuil, until_no_error)
        elif mode == "regression":
            return self.regression_train(training_data, seuil)
        else:
            print("Invalid mode")

import numpy as np
from typing import List

from utils.history import History
from multilayer.layer import Layer


class Model:
    """
    A class that represents a feedforward neural network model. It is made up of one or more layers.
    """

    def __init__(self, layers: List["Layer"]) -> None:
        """
        Initializes the model.
        :param layers: A list of Layer objects.
        """
        self.layers = layers
        self._connect_layers()
        self._evaluation = None

    def __repr__(self) -> str:
        """
        Returns a string representation of the model.
        """
        repr_str = "Model(\n"
        for i, layer in enumerate(self.layers):
            repr_str += f"  Layer {i}: {layer.units} neurons, activation={layer.activation_name}\n"
        repr_str += ")"
        return repr_str

    def _connect_layers(self) -> None:
        """
        Ensures all layers are properly connected (input sizes initialized).
        """
        for i in range(1, len(self.layers)):
            if self.layers[i].input_size is None:
                self.layers[i].set_input_size_from_previous_layer(self.layers[i - 1].units)

    def forward(self, input_vector: np.ndarray) -> np.ndarray:
        """
        Performs a forward pass through the entire model.
        :param input_vector: Input data to the first layer.
        :return: Final output of the network.
        """
        output = input_vector
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def retropropagation(self, error: np.ndarray, learning_rate: float) -> None:
        """
        Performs a backward pass through the entire model.
        :param error: Calculated error.
        :param learning_rate: Learning rate.
        """
        for layer in reversed(self.layers):
            error = layer.retropropagation(error, learning_rate)

    def _evaluate_full_batch(self, x_train: np.ndarray, y_train: np.ndarray, learning_rate: float, update: bool) -> tuple[np.ndarray,np.ndarray]:
        predictions = []
        errors = []

        for idx, iteration in enumerate(x_train):
            iteration = np.array(iteration)
            y_pred = self.forward(iteration)
            error = y_train[idx] - y_pred
            predictions.append(y_pred)
            errors.append(error)

        if update: # Not really a "full-batch" since we redo every one of them
            for idx, iteration in enumerate(x_train):
                iteration = np.array(iteration)
                y_pred = self.forward(iteration)
                error = y_train[idx] - y_pred
                self.retropropagation(error, learning_rate)

        return np.array(predictions), np.array(errors)

    def _evaluate_stochastic(self, x_train: np.ndarray, y_train: np.ndarray, learning_rate: float, update: bool) -> tuple[np.ndarray,np.ndarray]:
        predictions = []
        errors = []

        # Shuffle the data only when adjusting
        if update:
            indices = np.arange(x_train.shape[0])
            np.random.shuffle(indices)
            x_train = x_train[indices]
            y_train = y_train[indices]

        for idx, iteration in enumerate(x_train):
            iteration = np.array(iteration)
            y_pred = self.forward(iteration)
            error = y_train[idx] - y_pred

            if update:
                self.retropropagation(error, learning_rate)

            predictions.append(y_pred)
            errors.append(error)
        return np.array(predictions), np.array(errors)

    def _evaluate_mini_batch(self, x_train: np.ndarray, y_train: np.ndarray, learning_rate: float, update: bool) -> tuple[np.ndarray,np.ndarray]:
        pass

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, learning_rate: float, threshold: float, epochs:int, strategy: str="stochastic", verbose: bool=False) -> History:
        """
        Trains the model.
        :param x_train: Training data.
        :param y_train: Training labels.
        :param learning_rate: Learning rate.
        :param threshold: MSE threshold.
        :param epochs: Number of training epochs.
        :param strategy: Learning strategy.
        :param verbose: Verbosity.
        """
        history = History()
        strategy = strategy.lower()
        match strategy:
            case "full-batch":
                self._evaluation = self._evaluate_full_batch
            case "stochastic":
                self._evaluation = self._evaluate_stochastic
            case "mini-batch":
                raise NotImplementedError("Mini-batch evaluation not yet implemented.")
            case _:
                raise ValueError(f"Unknown learning strategy: {strategy}")
        print(f"Training using {strategy} strategy.")
        for epoch in range(epochs):
            self._evaluation(x_train, y_train, learning_rate, update=True)
            # Calculate MSE with new weights.
            predictions, errors = self._evaluation(x_train, y_train, learning_rate, update=False)
            mse = np.mean(0.5 * errors ** 2)
            predictions_rounded = predictions.round().reshape(y_train.shape)
            history.log(epoch=epoch, mse=mse, accuracy=np.mean(predictions_rounded == y_train))
            # Training exit condition
            if mse < threshold:
                print(f"Training complete after {epoch + 1} epochs.")
                return history
            if epoch % 1000 == 0 and verbose:
                print(f"Epoch :{epoch + 1}, MSE: {mse:.4f}")
        print(f"Training stopped after {epochs} epochs.")
        return history

    def save(self, path: str) -> None:
        """
        Saves the model to a file.
        :param path: Path to save the model to.
        """
        pass

    def load(self, path: str) -> None:
        """
        Loads the model from a file.
        :param path: Path to load the model from.
        """
        pass

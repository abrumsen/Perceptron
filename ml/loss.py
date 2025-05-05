import numpy as np

class Loss:
    """
    A wrapper class for loss functions and their gradients.
    """
    def __init__(self, name: str) -> None:
        """
        Initialize a loss function.
        :param name: The name of the loss function. Supported: "mse"
        """
        self.name = name.lower()

        match self.name:
            case "mse":
                self.func = self.mse
            case _:
                raise ValueError(f"Unsupported loss function: {name}")

    def __repr__(self) -> str:
        return f"Loss(name='{self.name}'"

    def __call__(self, y_true: np.ndarray | float, y_pred: np.ndarray | float) -> np.ndarray | float:
        return self.func(y_true, y_pred)

    @staticmethod
    def mse(y_true: np.ndarray | float, y_pred: np.ndarray | float) -> np.ndarray | float:
        """
        Mean Squared Error (MSE) loss function for classification.
        :param y_true: Ground truth labels.
        :param y_pred: Predicted values.
        :return: Mean squared error between y_true and y_pred.
        """
        return np.mean(0.5 * (y_true - y_pred) ** 2)
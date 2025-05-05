import numpy as np

class Activation:
    """
    A wrapper class for activation functions and their derivatives.
    """

    def __init__(self, name:str, **kwargs:float) -> None:
        """
        Initialize an activation function.
        :param name: The name of the activation function. Supported: "threshold", "identity", "sigmoid", "tanh"
        :param kwargs: Keyword arguments passed to the activation function.
        """
        self.name = name.lower()
        self.params = kwargs

        match self.name:
            case "threshold":
                self.func = self.threshold
                self.derivative = self.threshold_derivative
            case "identity":
                self.func = self.identity
                self.derivative = self.identity_derivative
            case "sigmoid":
                self.func = self.sigmoid
                self.derivative = self.sigmoid_derivative
            case "tanh":
                self.func = self.tanh
                self.derivative = self.tanh_derivative
            case _:
                raise ValueError(f"Unsupported activation function: {name}")

    def __repr__(self) -> str:
        return f"Activation(name='{self.name}', params={self.params})"

    def __call__(self, x: np.ndarray | float) -> np.ndarray | float:
        return self.func(x)

    def derivative_func(self, x: np.ndarray | float) -> np.ndarray | float:
        return self.derivative(x)

    @staticmethod
    def threshold(x:np.ndarray | float) -> np.ndarray | float:
            """
            Threshold function
            :param x: Calculated potential.
            :return: 1 if x >= 0, else 0.
            """
            return np.where(x >= 0, 1.0, 0.0)

    @staticmethod
    def identity(x:np.ndarray | float) -> np.ndarray | float:
            """
            Identity function
            :param x: Calculated potential.
            :return: x
            """
            return x

    def sigmoid(self, x: np.ndarray | float) -> np.ndarray | float:
        """
        Sigmoid function with adjustable slope
        :param x: Calculated potential.
        :return: sigmoid(x)
        """
        c = self.params.get("c", 1.0)
        return 1 / (1 + np.exp(-c * x))

    @staticmethod
    def tanh(x: np.ndarray | float) -> np.ndarray | float:
        """
        Tanh function
        :param x: Calculated potential.
        :return: tanh(x)
        """
        return np.tanh(x)

    def threshold_derivative(self, x:float) -> None:
        raise ValueError("Threshold function is not differentiable.")

    @staticmethod
    def identity_derivative(x:np.ndarray | float) -> np.ndarray | float:
        """
        Derivative of identity function
        :return: 1 for each element of the input, or a scalar 1 if input is a scalar.
        """
        return np.ones_like(x)

    def sigmoid_derivative(self, x: np.ndarray | float) -> np.ndarray | float:
        """
        Derivative of sigmoid function with adjustable slope
        :param x: Calculated potential.
        :return: slope * sigmoid(x) * (1 - sigmoid(x))
        """
        c = self.params.get("c", 1.0)
        s = self.sigmoid(x)
        return c * s * (1 - s)

    @staticmethod
    def tanh_derivative(x: np.ndarray | float) -> np.ndarray | float:
        """
        Derivative of tanh function
        :param x: Calculated potential.
        :return: 1 - tanh(x) ** 2
        """
        return 1 - np.tanh(x) ** 2

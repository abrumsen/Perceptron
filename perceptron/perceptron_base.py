import numpy as np

class Perceptron:
    def __init__(self, input_size: int, learning_rate: float = 0.3, epochs: int = 1000) -> None:
        if type(self) is Perceptron:
            raise RuntimeError("Perceptron is an abstract class and cannot be instantiated directly")
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = np.random.normal(loc=0.0, scale=1.0, size=input_size + 1)
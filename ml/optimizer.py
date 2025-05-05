class Optimizer:
    """
    A class that defines different learning algorithms and their methods.
    """
    def __init__(self, algorithm_name:str, learning_rate:float, mode:str) -> None:
        """
        Initializes the optimizer.
        """
        self.algorithm_name = algorithm_name.lower()
        self.learning_rate = learning_rate
        self.mode = mode

        match self.algorithm_name:
            case "simple":
                self.algorithm = self.simple
            case "gradient_descent":
                self.algorithm = self.gradient_descent
            case "adaline":
                self.algorithm = self.adaline
            case _:
                raise ValueError(f"Unsupported algorithm: {self.algorithm_name}")

        if self.mode == "regression" and self.algorithm == "simple":
            raise ValueError(f"Simple optimizer only supports 'classification' mode")

    def __repr__(self) -> str:
        return f"Optimizer(algorithm={self.algorithm_name}, learning_rate={self.learning_rate}, mode={self.mode})"

    def __call__(self):
        return self.algorithm()

    def simple(self):
        """
        Performs simple optimization.
        """
        pass
    def gradient_descent(self):
        """
        Performs gradient descent optimization.
        """
        pass
    def adaline(self):
        """
        Performs adaline optimization.
        """
        pass
    pass
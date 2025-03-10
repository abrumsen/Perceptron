import numpy as np
import pandas as pd
import pytest

from perceptron import PerceptronGradient


@pytest.fixture
def perceptron_gradient() -> PerceptronGradient:
    perceptron_gradient = PerceptronGradient(input_size=2, learning_rate=0.25, epochs=10000)
    perceptron_gradient.weights = np.array([0.0, 0.0, 0.0])
    return perceptron_gradient


def test_train_and_gate(perceptron_gradient: PerceptronGradient) -> None:
    """
    Test based on Table 2.2
    """
    training_data = pd.DataFrame({
        "inputs": [np.array([1, 0, 0]), np.array([1, 0, 1]), np.array([1, 1, 0]), np.array([1, 1, 1])],
        "label": [-1, -1, -1, 1]
    })
    result = perceptron_gradient.train(training_data=training_data, seuil=0.125001)
    assert result == 1
    assert np.array_equal(np.round(perceptron_gradient.weights,1), np.array([-1.5, 1.0, 1.0]))



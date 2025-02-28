import numpy as np
import pandas as pd
import pytest

from perceptron import PerceptronSimple


@pytest.fixture
def perceptron_simple() -> PerceptronSimple:
    perceptron_simple = PerceptronSimple(input_size=2, learning_rate=1, epochs=1000)
    perceptron_simple.weights = np.array([0, 0, 0])
    return perceptron_simple


def test_train_and_gate(perceptron_simple: PerceptronSimple) -> None:
    """
    Test based on Table 2.2
    """
    training_data = pd.DataFrame({
        "inputs": [np.array([1, 0, 0]), np.array([1, 0, 1]), np.array([1, 1, 0]), np.array([1, 1, 1])],
        "label": [0, 0, 0, 1]
    })
    result = perceptron_simple.train(training_data=training_data)
    assert result == 1
    assert np.array_equal(perceptron_simple.weights, np.array([-3, 2, 1]))

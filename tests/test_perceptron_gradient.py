import numpy as np
import pandas as pd
import pytest

from perceptron import PerceptronGradient


@pytest.fixture
def perceptron_gradient_classification() -> PerceptronGradient:
    perceptron_gradient = PerceptronGradient(input_size=2, learning_rate=0.2, epochs=1000)
    perceptron_gradient.weights = np.array([0.0, 0.0, 0.0])
    return perceptron_gradient

@pytest.fixture
def perceptron_gradient_regression() -> PerceptronGradient:
    perceptron_gradient = PerceptronGradient(input_size=2, learning_rate=0.000167, epochs=10000)
    perceptron_gradient.weights = np.array([0.0, 0.0])
    return perceptron_gradient


def test_train_and_gate_classification(perceptron_gradient_classification: PerceptronGradient) -> None:
    """
    Test based on Table 2.2
    """
    training_data = pd.DataFrame({
        "inputs": [np.array([1, 0, 0]), np.array([1, 0, 1]), np.array([1, 1, 0]), np.array([1, 1, 1])],
        "label": [-1, -1, -1, 1]
    })
    result = perceptron_gradient_classification.mode_choose(training_data=training_data, seuil=0.125001, mode=True)
    assert result == 1
    assert np.array_equal(np.round(perceptron_gradient_classification.weights,1), np.array([-1.5, 1.0, 1.0]))


def test_train_and_gate_regression(perceptron_gradient_regression: PerceptronGradient) -> None:
    """
    Test based on Table 1.30
    """
    training_data = pd.DataFrame({
        "inputs": [np.array([1, 10]), np.array([1, 14]), np.array([1, 12]), np.array([1, 18]), np.array([1, 16]),
                   np.array([1, 14]), np.array([1, 22]), np.array([1, 28]), np.array([1, 26]), np.array([1, 16]),
                   np.array([1, 23]), np.array([1, 25]), np.array([1, 20]), np.array([1, 20]), np.array([1, 24]),
                   np.array([1, 12]), np.array([1, 15]), np.array([1, 18]), np.array([1, 14]), np.array([1, 26]),
                   np.array([1, 25]), np.array([1, 17]), np.array([1, 12]), np.array([1, 20]), np.array([1, 23]),
                   np.array([1, 22]), np.array([1, 26]), np.array([1, 22]), np.array([1, 18]), np.array([1, 21]),
                   ],
        "label": [4.4, 5.6, 4.6, 6.1, 6.0, 7.0, 6.8, 10.6, 11.0, 7.6,
                  10.8, 10.0, 6.5, 8.2, 8.8, 5.5, 5.0, 8.0, 7.8, 9.0,
                  9.4, 8.5, 6.4, 7.5, 9.0, 8.1, 8.2, 10.0, 9.1, 9.0
                  ]
    })
    result = perceptron_gradient_regression.mode_choose(training_data=training_data, seuil=0.56, mode=False)
    assert result == 1
    assert np.array_equal(np.round(perceptron_gradient_regression.weights,2), np.array([1.57, 0.32]))

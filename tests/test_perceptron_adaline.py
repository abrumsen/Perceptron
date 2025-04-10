import numpy as np
import pandas as pd

from perceptron.perceptron_adaline import PerceptronAdaline

def perceptron_adaline() -> PerceptronAdaline:
    perceptron_adaline = PerceptronAdaline(input_size=2, learning_rate=0.03, epochs=10000)
    perceptron_adaline.weights = np.array([0.0,0.0,0.0])
    return perceptron_adaline

def train_adaline(perceptron_adaline: PerceptronAdaline) -> None:
    training_dataset = pd.DataFrame({
        "inputs": [np.array([1,0,0]), np.array([1,0,1]), np.array([1,1,0]), np.array([1,1,1])],
        "label": [-1, -1, -1, 1],
    })
    result = perceptron_adaline.train(training_dataset, seuil=0.1251)
    print(result)


def main():
    train_adaline(perceptron_adaline())

if __name__ == '__main__':
    main()

'''
Je vous prie de m'excuser mes seigneurs, je n'ai pas encore eu le temps de faire des tests super cool comme vous :c 
'''
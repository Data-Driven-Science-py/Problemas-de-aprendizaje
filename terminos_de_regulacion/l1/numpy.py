import numpy as np
from numpy.typing import NDArray

def sample_criterion(input: NDArray, target: NDArray) -> NDArray:
    return np.power(input - target, 2).mean()

def l1_regularization(weights: NDArray, lambd: float) -> NDArray:
    return lambd * np.abs(weights).sum()

def total_criterion(input: NDArray, target: NDArray, weights: NDArray, lambd: float) -> NDArray:
    return sample_criterion(input, target) + l1_regularization(weights, lambd)

if __name__ == '__main__':
    input: NDArray = np.random.randn(32, 10)
    target: NDArray = np.random.randn(32)
    weights: NDArray = np.random.randn(10, 1)
    lambd: float = 1e-3

    ## Ejemplo de propagacion: input @ weights
    out = total_criterion(input, target, weights, lambd)

    print('Array:', out)
    print('Media:', out.mean())
    print('Desviacion estandar:', out.std())


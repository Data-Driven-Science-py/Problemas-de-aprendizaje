"""
Estandares del codigo:

Pesos y parametros aprendibles: none
Valores de entrada: X
Valores de salida: O
"""
import numpy as np
from numpy.typing import NDArray

def dropout(X: NDArray, p: float = 0.5) -> NDArray:
    pShape: NDArray = np.random.rand(*X.shape) < p
    O: NDArray = X * pShape
    return O

if __name__ == '__main__':
    X: NDArray = np.random.randn(32, 10)

    Y = dropout(X)

    print('Array:', Y)
    print('Media:', Y.mean())
    print('Desviacion estandar:', Y.std())


"""
Estandares del codigo:

Pesos y parametros aprendibles: beta, gamma
Valores de entrada: X, x
Valores de salida: Y, y
Media: mu
Varianza: sig2
Desviacion estandar: sigma
"""
import numpy as np
from numpy.typing import NDArray

def normalizacion_por_bache(x: NDArray, beta: float, gamma: float, epsilon: float = 1e-6) -> NDArray:
    ## 1. Sacar la media y varianza de los baches
    ## (axis = 0) indica que estamos tomando la medida desde la dimension de los baches
    mu: NDArray = x.mean(axis = 0)
    var: NDArray = x.var(axis = 0)
    ## 2. Normalizamos de manera estandar
    x_hat: NDArray = (x - mu) / np.sqrt(var + epsilon)
    ## 3. Escalar el valor normalizado con el peso aprendible gamma, y desviar el centro de los datos con el bias beta
    y: NDArray = gamma * x_hat + beta
    return y

if __name__ == '__main__':
    ## Podemos ir probando los valores para ver que ocurre con una distribucion normal al variar los parametros

    x: NDArray = np.random.randn(32, 10)
    beta: float = 5.
    gamma: float = 2.

    out = normalizacion_por_bache(x, beta, gamma)

    print('Array:', out)
    print('Media:', out.mean())
    print('Desviacion estandar:', out.std())


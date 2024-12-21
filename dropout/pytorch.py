from torch.nn import Dropout, Linear, Sequential, ReLU

## Implementacion de la red neuronal de la figura 1.

class RedNeuronalDropout(Sequential):
    def __init__(self, input_size: int, out_size: int) -> None:
        super().__init__(
            Linear(input_size, 4),
            ReLU(),
            Linear(4, 16),
            ReLU(),
            Linear(16, 32),
            ReLU(),
            Dropout(p = 0.5), ## this is the torch implementation for the dropout, p is the probability of dropping each perceptron, randomly sampled.
            Linear(32, 20),
            Dropout(p = 0.5),
            ReLU(),
            Linear(20, out_size)

        )

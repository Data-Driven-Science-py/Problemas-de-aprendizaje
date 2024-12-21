from torch.nn import LayerNorm, Linear, Sequential, Conv2d, BatchNorm1d, ReLU, Sigmoid, MaxPool2d, Flatten

## Implementacion de la red neuronal de la figura 1.

class RedConvolucional(Sequential):
    def __init__(self, in_channels: int) -> None:
        super().__init__(
            Conv2d(in_channels, out_channels= 16, kernel_size= (3, 3), stride = 1, padding = 1),
            ReLU(),
            MaxPool2d(2, 2),
            LayerNorm(16),
            Conv2d(16, 32, (3, 3), 1, 1),
            ReLU(),
            MaxPool2d(2, 2),
            LayerNorm(32),
            Conv2d(32, 64, (3, 3), 1, 1),
            ReLU(),
            MaxPool2d(2, 2),
            LayerNorm(64),
            Flatten()
        )

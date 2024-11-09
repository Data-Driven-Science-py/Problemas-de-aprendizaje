from torch.nn import BatchNorm2d, Linear, Sequential, Conv2d, BatchNorm1d, ReLU, Sigmoid, MaxPool2d, Flatten

## Implementacion de la red neuronal de la figura 1.

class RedNeuronal(Sequential):
    def __init__(self, input_size: int) -> None:
        super().__init__(
            Linear(input_size, 4),
            ReLU(),
            BatchNorm1d(4),
            Linear(4, 16),
            ReLU(),
            BatchNorm1d(16),
            Linear(16, 32),
            ReLU(),
            BatchNorm1d(32),
            Linear(32, 12),
            ReLU(),
            BatchNorm1d(12),
            Linear(12, 5),
            ReLU(),
            BatchNorm1d(5),
            Linear(5, 1),
            Sigmoid()
        )

class RedConvolucional(Sequential):
    def __init__(self, in_channels: int) -> None:
        super().__init__(
            Conv2d(in_channels, out_channels= 16, kernel_size= (3, 3), stride = 1, padding = 1),
            ReLU(),
            MaxPool2d(2, 2),
            BatchNorm2d(num_features = 16),
            Conv2d(16, 32, (3, 3), 1, 1),
            ReLU(),
            MaxPool2d(2, 2),
            BatchNorm2d(32),
            Conv2d(32, 64, (3, 3), 1, 1),
            ReLU(),
            MaxPool2d(2, 2),
            BatchNorm2d(64),
            Flatten()
        )

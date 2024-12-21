from torch.nn import InstanceNorm2d, Sequential, Conv2d,ReLU, MaxPool2d, Flatten

class RedConvolucional(Sequential):
    def __init__(self, in_channels: int) -> None:
        super().__init__(
            Conv2d(in_channels, out_channels= 16, kernel_size= (3, 3), stride = 1, padding = 1),
            ReLU(),
            MaxPool2d(2, 2),
            InstanceNorm2d(16),
            Conv2d(16, 32, (3, 3), 1, 1),
            ReLU(),
            MaxPool2d(2, 2),
            InstanceNorm2d(32),
            Conv2d(32, 64, (3, 3), 1, 1),
            ReLU(),
            MaxPool2d(2, 2),
            InstanceNorm2d(64),
            Flatten()
        )

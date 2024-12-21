from typing import List
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Sigmoid
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn.functional as f
from torch import Tensor
from tqdm import tqdm ## esto es para poder mostrar el proceso de entranamiento

def criterion(x: Tensor, target: Tensor) -> Tensor:
    return f.binary_cross_entropy(x, target)

def get_optimizer(model, lr: float, lambd: float):
    return Adam(model.parameters(), lr = lr, weight_decay= lambd) ## weight_decay es el lambda para la implementacion de pytorch de la regularizacion l2

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

def train(
    epochs: int, lr: float, lambd: float,
    train_loader: DataLoader, val_loader: DataLoader,
) -> None:
    model = RedNeuronal(10)
    optimizer = get_optimizer(model, lr, lambd)
    for epoch in range(epochs):
        train_history: List = []
        for batch in tqdm(train_loader, desc = f"Training - Epoch: {epoch}"):
            x, y = batch
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(loss.detach().item())
        print(f"Training - Epoch {epoch} loss: ", sum(train_history) / len(train_history))

        val_history: List = []
        for batch in tqdm(val_loader, desc = f"Validation - Epoch: {epoch}"):
            x, y = batch
            y_pred = model(x)
            loss = criterion(y_pred, y)
            val_history.append(loss.detach().item())
        print(f"Validation - Epoch {epoch} loss: ", sum(val_history) / len(val_history))




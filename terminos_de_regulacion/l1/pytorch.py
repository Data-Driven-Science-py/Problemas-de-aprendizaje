import torch

## Implementacion de la red neuronal de la figura 1.
## No hay un metodo construido dentro de pytorch para l1, pero podemos recrearlo de la siguiente manera:
def regularizacion_l1(model, lambd: float):
    loss = 0.
    for param in model.parameters():
        loss += torch.abs(param).sum()
    return loss * lambd

## Y esto luego agregar a la perdida
## loss = mse(input, target) + regularizacion_l1(model, lambd)

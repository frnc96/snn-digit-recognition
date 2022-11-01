from __future__ import annotations
import torch as T
from torch.utils.data import DataLoader


def evaluate_loss_dataloader(nets: list[T.nn.Module], data_loader: DataLoader, device: str) -> list[tuple[float, T.nn.Module]]:
    # Set the loss function
    loss_function = T.nn.CrossEntropyLoss()

    for net in nets:
        net.eval()

    counter = 0
    results = [[[], net] for net in nets]
    with T.no_grad():
        # Loop through data in batches
        for data, targets in data_loader:
            counter += 1

            # Forward pass
            data = data.to(device)
            targets = targets.to(device)
            for i, net in enumerate(nets):
                y_pred = net(data)
                results[i][0].append(loss_function(y_pred, targets).item())

    # TODO: Weight loss function to account for batch size.
    return [(T.Tensor(losses).mean().item(), net) for losses, net in results]


def evaluate_loss_tensor(net: T.nn.Module, X: T.Tensor, y: T.Tensor, device: str) -> float:
    y_pred = net(X)
    loss_function = T.nn.CrossEntropyLoss()
    return loss_function(y_pred, y).item()


def evaluate_accuracy_dataloader(net: T.nn.Module, data_loader: DataLoader, device: str) -> float:
    total = 0
    correct = 0

    # Iterate through all mini-batches and measure accuracy over the full data set
    net.eval()
    with T.no_grad():
        for data, targets in data_loader:
            data = data.to(device)
            targets = targets.to(device)

            # forward pass
            y_pred = net(data)

            # calculate total accuracy
            # The output and target are both of size (N, 10), so argmax is used to get the
            # proper prediction.
            y_pred = T.argmax(y_pred, dim=1)
            targets = T.argmax(targets, dim=1)
            correct += (y_pred == targets).sum().item()
            total += targets.size(0)

    return correct / total


def evaluate_accuracy_tensor(net: T.nn.Module, X: T.Tensor, y: T.Tensor, device: str) -> float:
    y_pred = net(X)
    y_pred = T.argmax(y_pred, dim=1)
    y = T.argmax(y, dim=1)
    return (y_pred == y).sum().item() / y.size(0)

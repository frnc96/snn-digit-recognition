from __future__ import annotations

from dataclasses import dataclass

import torch as T
from torch.utils.data import DataLoader


@dataclass
class EvaluationResult:
    module: T.nn.Module
    value: float


def evaluate_loss_dataloader(
    nets: list[T.nn.Module],
    data_loader: DataLoader,
    device: str
) -> list[EvaluationResult]:
    """Evaluate loss on a data loader for a set of networks.

    Args:
        nets: List of networks to evaluate.
        data_loader: Data loader to evaluate on.
        device: Device to evaluate on.
    Returns:
        List of tuples of (loss, network) for each network.
    """
    loss_function = T.nn.CrossEntropyLoss()

    for net in nets:
        net.eval()

    counter = 0
    results: list[list[float]] = [[] for _ in nets]
    with T.no_grad():
        # Loop through data in batches
        for data, targets in data_loader:
            counter += 1

            # Forward pass
            data = data.to(device)
            targets = targets.to(device)
            for i, net in enumerate(nets):
                y_pred = net(data)
                results[i].append(float(loss_function(y_pred, targets).item()))

    # TODO: Weight loss function to account for batch size.
    return [
        EvaluationResult(value=T.Tensor(losses).mean().item(), module=net)
        for losses, net in zip(results, nets)
    ]


def evaluate_accuracy_dataloader(nets: list[T.nn.Module], data_loader: DataLoader, device: str) -> list[EvaluationResult]:
    """Evaluate accuracy on a data loader for a set of networks.

    Args:
        nets: List of networks to evaluate.
        data_loader: Data loader to evaluate on.
        device: Device to evaluate on.
    Returns:
        List of accuracies for each network.
    """

    for net in nets:
        net.eval()

    total = 0
    results: list[int] = [0 for _ in nets]
    with T.no_grad():
        for data, targets in data_loader:
            data = data.to(device)
            targets = targets.to(device)
            targets = T.argmax(targets, dim=1)
            total += targets.size(0)

            for i, net in enumerate(nets):
                # calculate total accuracy
                # The output and target are both of size (N, 10), so argmax is used to get the
                # proper prediction.
                y_pred = net(data)
                y_pred = T.argmax(y_pred, dim=1)
                results[i] += int((y_pred == targets).sum().cpu().item())

    return [
        EvaluationResult(value=correct / total, module=net)
        for correct, net in zip(results, nets)
    ]


def evaluate_loss_tensor(net: T.nn.Module, X: T.Tensor, y: T.Tensor, device: str) -> float:
    y_pred = net(X)
    loss_function = T.nn.CrossEntropyLoss()
    return loss_function(y_pred, y).item()


def evaluate_accuracy_tensor(net: T.nn.Module, X: T.Tensor, y: T.Tensor, device: str) -> float:
    y_pred = net(X)
    y_pred = T.argmax(y_pred, dim=1)
    y = T.argmax(y, dim=1)
    return (y_pred == y).sum().item() / y.size(0)

from __future__ import annotations

import torch as T


def generate_mutation_tensor(size: T.Size, prob, std, device: str | T.device):
    """Generates a matrix of random values with a given probability of mutation.
    The matrix can be added to any other matrix of the same size to mutate it.

    Args:
        size (T.Size): The size of the matrix to generate.
        prob (float): The probability of mutation.
        std (float): The standard deviation of the mutation.
        device (str | T.device): The device to generate the matrix on.
    Returns:
        T.Tensor: The mutation matrix.
    """
    # Parameters for the normal function.
    mean = T.zeros(size, device=device)
    std = T.full(size, std, device=device)
    chance = T.full(size, prob, device=device)
    bernoulli = T.bernoulli(chance)

    # Generate gaussian noise based on the generated parameters.
    pertub = T.normal(mean=mean, std=std)

    return pertub * bernoulli


def tensor_crossover(t1: T.Tensor, t2: T.Tensor):
    """Creates a new tensor by randomly selecting values from two parent tensors
    using 1-point crossover.

    Args:
        t1 (T.Tensor): The first tensor.
        t2 (T.Tensor): The second tensor.
    Returns:
        T.Tensor: The crossover tensor.
    """
    if not t1.size() == t2.size():
        raise ValueError("Tensor size mismatch")

    size = t1.size()
    t1 = T.flatten(t1.clone())
    t2 = T.flatten(t2.clone())
    cut_idx = int(T.rand(1).item() * t1.size(0))
    t1[cut_idx:] = 0
    t2[:cut_idx] = 0
    t3 = t1 + t2

    return T.reshape(t3, size)

from __future__ import annotations

import torch as T


def generate_mutation_matrix(size: T.Size, prob, std, device: str | T.device):
    mean = T.zeros(size, device=device)

    std = T.full(size, std, device=device)
    chance = T.full(size, prob, device=device)
    bernoulli = T.bernoulli(chance)
    pertub = T.normal(mean=mean, std=std)

    return pertub * bernoulli


def tensor_crossover(t1: T.Tensor, t2: T.Tensor):
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

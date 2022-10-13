import torch
import random
import src.domain.constants.parameters as params


class EvoHelpers:

    @staticmethod
    def generate_mutation_matrix(size, prob, std, device):
        mean = torch.zeros(size, device=device)

        std = torch.full(size, std, device=device)
        chance = torch.full(size, prob, device=device)
        bernoulli = torch.bernoulli(chance)
        pertub = torch.normal(mean=mean, std=std)

        return pertub * bernoulli

    @staticmethod
    def tensor_crossover(t1: torch.Tensor, t2: torch.Tensor):
        if not t1.size() == t2.size():
            raise ValueError("Tensor size mismatch")

        size = t1.size()
        t1 = torch.flatten(t1.clone())
        t2 = torch.flatten(t2.clone())
        cut_idx = int(torch.rand(1).item() * t1.size(0))
        t1[cut_idx:] = 0
        t2[:cut_idx] = 0
        t3 = t1 + t2

        return torch.reshape(t3, size)

    @staticmethod
    def evaluate_loss(model, X, y, n_samples=None, with_replacement=False):
        # todo - implement this
        return random.randint(0, 5)
        # if n_samples:
        #     if with_replacement:
        #         idxs = torch.randint(len(X), (n_samples,))
        #     else:
        #         idxs = torch.randperm(X.size(0))[:n_samples]
        #
        #     X, y = X[idxs].to(params.DEVICE), y[idxs].to(params.DEVICE)
        # else:
        #     X, y = X.to(params.DEVICE), y.to(params.DEVICE)
        #
        # return torch.nn.functional.cross_entropy(model(X), y).item()

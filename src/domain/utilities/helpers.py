import time
import torch
from torch import nn
import src.domain.constants.parameters as params

dtype = torch.float


class EvoHelpers:

    @staticmethod
    def generate_mutation_matrix(size, prob, std):
        mean = torch.zeros(size, device=params.DEVICE)

        std = torch.full(size, std, device=params.DEVICE)
        chance = torch.full(size, prob, device=params.DEVICE)
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
    def evaluate_loss(net, data_loader, counter):
        start_time = time.time()
        data_batch = iter(data_loader)

        # Set the loss function
        loss_function = nn.CrossEntropyLoss()
        loss_history = []

        # Loop through data in batches
        for data, targets in data_batch:
            # Forward pass
            net.train()
            spk_rec, mem_rec = net(data.view(params.BATCH_SIZE, -1))

            # Initialize the loss & sum over time
            loss_val = torch.zeros(1, dtype=dtype, device=params.DEVICE)
            for step in range(params.NUM_OF_STEPS):
                loss_val += loss_function(mem_rec[step], targets)

            # Record the loss
            loss_history.append(loss_val.item() / params.NUM_OF_STEPS)

        print(f"Calculated loss for agent {counter}/{params.POPULATION_SIZE} in {time.time() - start_time:.2f} seconds")

        # Return the avg loss value
        return sum(loss_history) / len(loss_history)

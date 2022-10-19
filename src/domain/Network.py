import torch
import torch.nn as nn
import snntorch as snn
import src.domain.constants.parameters as params
from src.domain.utilities.helpers import EvoHelpers


# Define Network
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # Initialize SNN layers
        self.fc1 = nn.Linear(params.NUM_OF_INPUT_NEURONS, params.NUM_OF_HIDDEN_NEURONS)
        self.lif1 = snn.Leaky(beta=params.LEAKY_BETA)
        self.fc2 = nn.Linear(params.NUM_OF_HIDDEN_NEURONS, params.NUM_OF_OUTPUT_NEURONS)
        self.lif2 = snn.Leaky(beta=params.LEAKY_BETA)

        self.mutation_rate = nn.Parameter(torch.Tensor([params.MUTATION_RATE]), requires_grad=False)
        self.mutation_std = nn.Parameter(torch.Tensor([params.MUTATION_STD]), requires_grad=False)

        self.linear_layers = [
            self.fc1,
            self.fc2,
        ]

    def forward(self, x):
        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # Record the final layer
        spk2_rec = []
        mem2_rec = []

        for step in range(params.NUM_OF_STEPS):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)

    def mutate(self):
        with torch.no_grad():
            self.mutation_rate += torch.normal(mean=0, std=torch.Tensor([0.01]).to(params.DEVICE))
            self.mutation_rate[0] = torch.clip(self.mutation_rate, min=0.001, max=1)[0].item()
            self.mutation_std += torch.normal(mean=0, std=torch.Tensor([0.01]).to(params.DEVICE))
            self.mutation_std[0] = torch.clip(self.mutation_std, min=0.001, max=3)[0].item()

            mut_rate = self.mutation_rate[0].item()
            mut_std = self.mutation_std[0].item()

            for layer in self.linear_layers:
                layer.weight += EvoHelpers.generate_mutation_matrix(layer.weight.size(), mut_rate, mut_std)
                layer.bias += EvoHelpers.generate_mutation_matrix(layer.bias.size(), mut_rate, mut_std)

        return self

    def crossover(self, other):
        child = Net().to(params.DEVICE)
        child.load_state_dict(self.state_dict())

        with torch.no_grad():
            for layer_new, layer_p1, layer_p2 in zip(child.linear_layers, self.linear_layers, other.linear_layers):
                layer_new.weight[:, :] = EvoHelpers.tensor_crossover(layer_p1.weight, layer_p2.weight)
                layer_new.bias[:] = EvoHelpers.tensor_crossover(layer_p1.bias, layer_p2.bias)

        return child

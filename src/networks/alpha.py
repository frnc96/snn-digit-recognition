import snntorch as snn
import torch as T
import torch.nn as nn
from src.networks.helpers import generate_mutation_matrix, tensor_crossover
from src.networks.base_network import BaseNetwork
from copy import deepcopy


class AlphaBase(nn.Module, BaseNetwork):
    def __init__(
        self,
        num_inputs: int = 28*28,
        num_outputs: int = 10,
        alpha: float = 0.2,
        beta: float = 0.1,
        threshold: float = 1.0,
    ):
        super().__init__()
        self.leaky_beta = beta
        self.leaky_threshold = threshold
        self.softmax = T.nn.Softmax(dim=1)

        # Initialize SNN layers
        self.fc1 = nn.Linear(num_inputs, num_outputs)
        self.alpha1 = snn.Alpha(alpha=alpha, beta=beta, threshold=threshold)

        self.dev = T.nn.parameter.Parameter(T.empty([1]), requires_grad=False)
        self.mutation_rate = nn.parameter.Parameter(T.Tensor([0.5]), requires_grad=False)
        self.mutation_std = nn.parameter.Parameter(T.Tensor([0.5]), requires_grad=False)

        self.layers = [
            self.fc1,
            self.alpha1,
        ]

    def describe(self) -> str:
        layers = []
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                layers.append(f"Linear: Weights.size() = {layer.weight.size()}")
            elif isinstance(layer, snn.Alpha):
                layers.append(f"Alpha: Alpha.size() = {layer.alpha.size()}, Beta.size() = {layer.beta.size()}")
        return "Network(\n    " + ",\n    ".join(layers) + "\n)"

    def mutate(self) -> 'AlphaBase':
        with T.no_grad():
            self.mutation_rate += T.normal(mean=0, std=T.Tensor([0.01]).to(self.dev.device))
            self.mutation_rate.data = T.clip(self.mutation_rate.data, min=0.001, max=1)
            self.mutation_std += T.normal(mean=0, std=T.Tensor([0.01]).to(self.dev.device))
            self.mutation_std.data = T.clip(self.mutation_std.data, min=0.001, max=3)

            mut_rate = self.mutation_rate[0].item()
            mut_std = self.mutation_std[0].item()

            for layer in self.layers:
                if isinstance(layer, nn.Linear):
                    layer.weight.data += generate_mutation_matrix(layer.weight.size(), mut_rate, mut_std, device=self.dev.device)
                    layer.bias.data += generate_mutation_matrix(layer.bias.size(), mut_rate, mut_std, device=self.dev.device)
                elif isinstance(layer, snn.Alpha):
                    pass
                    # layer.alpha.data += generate_mutation_matrix(layer.alpha.size(), mut_rate, mut_std, device=self.dev.device)
                    # layer.alpha.data = T.clip(layer.alpha.data, min=0.0, max=1.0)
                    # layer.beta.data += generate_mutation_matrix(layer.beta.size(), mut_rate, mut_std, device=self.dev.device)
                    # layer.beta.data = T.clip(layer.beta.data, min=0.0, max=1.0)
                    # layer.threshold.data += generate_mutation_matrix(layer.threshold.size(), mut_rate, mut_std, device=self.dev.device)
                    # layer.threshold.data = T.clip(layer.threshold.data, min=0.0, max=10.0)

        return self

    def crossover(self, other: 'AlphaBase') -> 'AlphaBase':
        child = deepcopy(self).to(self.dev.device)
        child.load_state_dict(self.state_dict())

        with T.no_grad():
            child.mutation_rate.data = (self.mutation_rate.data + other.mutation_rate.data) / 2
            child.mutation_std.data = (self.mutation_std.data + other.mutation_std.data) / 2
            for layer_new, layer_p1, layer_p2 in zip(child.layers, self.layers, other.layers):
                if isinstance(layer_new, nn.Linear):
                    layer_new.weight.data = tensor_crossover(layer_p1.weight, layer_p2.weight)
                    layer_new.bias.data = tensor_crossover(layer_p1.bias, layer_p2.bias)
                elif isinstance(layer_new, snn.Alpha):
                    pass
                    # layer_new.threshold.data = (layer_p1.threshold.data + layer_p2.threshold.data) / 2
                    # if len(layer_new.alpha.data.size()) > 0:
                    #     layer_new.alpha.data = tensor_crossover(layer_p1.alpha, layer_p2.alpha)
                    # else:
                    #     layer_new.alpha.data = (layer_p1.alpha.data + layer_p2.alpha.data) / 2
                    # if len(layer_new.beta.data.size()) > 0:
                    #     layer_new.beta.data = tensor_crossover(layer_p1.beta, layer_p2.beta)
                    # else:
                    #     layer_new.beta.data = (layer_p1.beta.data + layer_p2.beta.data) / 2

        return child


class AlphaOneLayer(AlphaBase):
    def __init__(self):
        super().__init__()

    def describe(self) -> str:
        layers = []
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                layers.append(f"Linear: Weights.size() = {layer.weight.size()}")
            elif isinstance(layer, snn.Leaky):
                layers.append(f"Leaky: Beta.size() = {layer.beta.size()}")
        return "Network(\n    " + ",\n    ".join(layers) + "\n)"

    def forward(self, x):
        # Initialize hidden states at t=0
        syn_exc1, syn_inh1, mem1 = self.alpha1.init_alpha()

        # Record the final layer
        spk_rec = []

        for step in range(x.size(1)):
            cur1 = self.fc1(x[:, step, :])
            spk1, syn_exc1, syn_inh1, mem1 = self.alpha1(cur1, syn_exc1, syn_inh1, mem1)
            spk_rec.append(spk1)

        y = T.stack(spk_rec, dim=1)
        y = y.sum(dim=1)

        y = self.softmax(y)

        return y


class AlphaOneLayer_NoAlphaBetaEvolution_Alpha60pct_Beta40pct(AlphaBase):
    def __init__(self):
        super().__init__()

    def describe(self) -> str:
        layers = []
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                layers.append(f"Linear: Weights.size() = {layer.weight.size()}")
            elif isinstance(layer, snn.Leaky):
                layers.append(f"Leaky: Beta.size() = {layer.beta.size()}")
        return "Network(\n    " + ",\n    ".join(layers) + "\n)"

    def forward(self, x):
        # Initialize hidden states at t=0
        syn_exc1, syn_inh1, mem1 = self.alpha1.init_alpha()

        # Record the final layer
        spk_rec = []

        for step in range(x.size(1)):
            cur1 = self.fc1(x[:, step, :])
            spk1, syn_exc1, syn_inh1, mem1 = self.alpha1(cur1, syn_exc1, syn_inh1, mem1)
            spk_rec.append(spk1)

        y = T.stack(spk_rec, dim=1)
        y = y.sum(dim=1)

        y = self.softmax(y)

        return y

    def mutate(self) -> 'AlphaBase':
        with T.no_grad():
            self.mutation_rate += T.normal(mean=0, std=T.Tensor([0.01]).to(self.dev.device))
            self.mutation_rate.data = T.clip(self.mutation_rate.data, min=0.001, max=1)
            self.mutation_std += T.normal(mean=0, std=T.Tensor([0.01]).to(self.dev.device))
            self.mutation_std.data = T.clip(self.mutation_std.data, min=0.001, max=3)

            mut_rate = self.mutation_rate[0].item()
            mut_std = self.mutation_std[0].item()

            for layer in self.layers:
                if isinstance(layer, nn.Linear):
                    layer.weight.data += generate_mutation_matrix(layer.weight.size(), mut_rate, mut_std, device=self.dev.device)
                    layer.bias.data += generate_mutation_matrix(layer.bias.size(), mut_rate, mut_std, device=self.dev.device)
                elif isinstance(layer, snn.Alpha):
                    layer.threshold.data += generate_mutation_matrix(layer.threshold.size(), mut_rate, mut_std, device=self.dev.device)
                    layer.threshold.data = T.clip(layer.threshold.data, min=0.0, max=10.0)

        return self

    def crossover(self, other: 'AlphaBase') -> 'AlphaBase':
        child = deepcopy(self).to(self.dev.device)
        child.load_state_dict(self.state_dict())

        with T.no_grad():
            child.mutation_rate.data = (self.mutation_rate.data + other.mutation_rate.data) / 2
            child.mutation_std.data = (self.mutation_std.data + other.mutation_std.data) / 2
            for layer_new, layer_p1, layer_p2 in zip(child.layers, self.layers, other.layers):
                if isinstance(layer_new, nn.Linear):
                    layer_new.weight.data = tensor_crossover(layer_p1.weight, layer_p2.weight)
                    layer_new.bias.data = tensor_crossover(layer_p1.bias, layer_p2.bias)
                elif isinstance(layer_new, snn.Alpha):
                    layer_new.threshold.data = (layer_p1.threshold.data + layer_p2.threshold.data) / 2

        return child

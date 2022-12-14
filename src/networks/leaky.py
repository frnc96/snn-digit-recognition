"""Networks implementing leaky neurons.

The network usage is specified in the class name and is
meant for ease of reproducibility.

Crossover and mutation functions are implemented to allow for
specific behavior that's easy to extend.
"""


from copy import deepcopy

import snntorch as snn
import torch as T
import torch.nn as nn

from src.networks.base_network import BaseNetwork
from src.networks.helpers import generate_mutation_tensor, tensor_crossover


class LeakyBase(nn.Module, BaseNetwork):
    def __init__(
        self,
        num_inputs: int = 28*28,
        num_outputs: int = 10,
        beta: float = 0.1,
        threshold: float = 1.0,
    ):
        super().__init__()

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.leaky_beta = beta
        self.leaky_threshold = threshold
        self.softmax = T.nn.Softmax(dim=1)

        # Initialize SNN layers
        self.fc1 = nn.Linear(num_inputs, num_outputs)
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold)

        self.dev = T.nn.parameter.Parameter(T.empty([1]), requires_grad=False)
        self.mutation_rate = nn.parameter.Parameter(T.Tensor([0.5]), requires_grad=False)
        self.mutation_std = nn.parameter.Parameter(T.Tensor([0.5]), requires_grad=False)
        self.mutation_std_0to1 = nn.parameter.Parameter(T.Tensor([0.0001]), requires_grad=False)

        self.layers = [
            self.fc1,
            self.lif1,
        ]

    def describe(self) -> str:
        layers = []
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                layers.append(f"Linear: Weights.size() = {layer.weight.size()}")
            elif isinstance(layer, snn.Leaky):
                layers.append(f"Leaky: Beta.size() = {layer.beta.size()}")
        return "Network(\n    " + ",\n    ".join(layers) + "\n)"

    def mutate(self) -> 'LeakyBase':
        with T.no_grad():
            self.mutation_rate += T.normal(mean=0, std=T.Tensor([0.01]).to(self.dev.device))
            self.mutation_rate.data = T.clip(self.mutation_rate.data, min=0.001, max=1)
            self.mutation_std += T.normal(mean=0, std=T.Tensor([0.01]).to(self.dev.device))
            self.mutation_std.data = T.clip(self.mutation_std.data, min=0.001, max=3)
            self.mutation_std_0to1 += T.normal(mean=0, std=T.Tensor([0.1]).to(self.dev.device))
            self.mutation_std_0to1.data = T.clip(self.mutation_std_0to1.data, min=0.00001, max=1.0)

            mut_rate = self.mutation_rate[0].item()
            mut_std = self.mutation_std[0].item()
            mut_std_0to1 = self.mutation_std_0to1[0].item()

            for layer in self.layers:
                if isinstance(layer, nn.Linear):
                    layer.weight.data += generate_mutation_tensor(layer.weight.size(), mut_rate, mut_std, device=self.dev.device)
                    layer.bias.data += generate_mutation_tensor(layer.bias.size(), mut_rate, mut_std, device=self.dev.device)
                elif isinstance(layer, snn.Leaky):
                    layer.beta.data += generate_mutation_tensor(layer.beta.size(), mut_rate, mut_std_0to1, device=self.dev.device)
                    layer.beta.data = T.clip(layer.beta.data, min=0.0, max=1.0)
                    layer.threshold.data += generate_mutation_tensor(layer.threshold.size(), mut_rate, mut_std, device=self.dev.device)
                    layer.threshold.data = T.clip(layer.threshold.data, min=0.0, max=10.0)

        return self

    def crossover(self, other: 'LeakyBase') -> 'LeakyBase':
        child = deepcopy(self).to(self.dev.device)
        child.load_state_dict(self.state_dict())

        with T.no_grad():
            child.mutation_rate.data = (self.mutation_rate.data + other.mutation_rate.data) / 2
            child.mutation_std.data = (self.mutation_std.data + other.mutation_std.data) / 2
            for layer_new, layer_p1, layer_p2 in zip(child.layers, self.layers, other.layers):
                if isinstance(layer_new, nn.Linear):
                    layer_new.weight.data = tensor_crossover(layer_p1.weight, layer_p2.weight)
                    layer_new.bias.data = tensor_crossover(layer_p1.bias, layer_p2.bias)
                elif isinstance(layer_new, snn.Leaky):
                    layer_new.threshold.data = (layer_p1.threshold.data + layer_p2.threshold.data) / 2
                    if len(layer_new.beta.data.size()) > 0:
                        layer_new.beta.data = tensor_crossover(layer_p1.beta, layer_p2.beta)
                    else:
                        layer_new.beta.data = (layer_p1.beta.data + layer_p2.beta.data) / 2

        return child


class LeakyOneLayer(LeakyBase):
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
        mem1 = self.lif1.init_leaky()

        # Record the final layer
        spk_rec = []

        for step in range(x.size(1)):
            cur1 = self.fc1(x[:, step, :])
            spk1, mem1 = self.lif1(cur1, mem1)
            spk_rec.append(spk1)

        y = T.stack(spk_rec, dim=1)
        y = y.sum(dim=1)

        y = self.softmax(y)

        return y


class LeakyTwoLayer(LeakyBase):
    def __init__(
        self,
        num_hidden: int = 10,
    ):
        super().__init__()

        # Initialize SNN layers
        self.fc1 = nn.Linear(self.num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=self.leaky_beta, threshold=self.leaky_threshold)
        self.fc2 = nn.Linear(num_hidden, self.num_outputs)
        self.lif2 = snn.Leaky(beta=self.leaky_beta, threshold=self.leaky_threshold)

        self.layers = [
            self.fc1,
            self.lif1,
            self.fc2,
            self.lif2,
        ]

    def forward(self, x):
        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # Record the final layer
        spk_rec = []

        for step in range(x.size(1)):
            cur1 = self.fc1(x[:, step, :])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk_rec.append(spk2)

        y = T.stack(spk_rec, dim=1)
        y = y.sum(dim=1)

        y = self.softmax(y)

        return y


class LeakyThreeLayer(LeakyBase):
    def __init__(
        self,
        num_hidden: int = 10,
    ):
        super().__init__()

        # Initialize SNN layers
        self.fc1 = nn.Linear(self.num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=self.leaky_beta, threshold=self.leaky_threshold)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.lif2 = snn.Leaky(beta=self.leaky_beta, threshold=self.leaky_threshold)
        self.fc3 = nn.Linear(num_hidden, self.num_outputs)
        self.lif3 = snn.Leaky(beta=self.leaky_beta, threshold=self.leaky_threshold)

        self.layers = [
            self.fc1,
            self.lif1,
            self.fc2,
            self.lif2,
            self.fc3,
            self.lif3,
        ]

    def forward(self, x):
        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        # Record the final layer
        spk_rec = []

        for step in range(x.size(1)):
            cur1 = self.fc1(x[:, step, :])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)
            spk_rec.append(spk3)

        y = T.stack(spk_rec, dim=1)
        y = y.sum(dim=1)

        y = self.softmax(y)

        return y


class LeakyOneLayer_NoBetaEvolution(LeakyBase):
    def __init__(self):
        super().__init__()

        self.lif1.beta.data = T.Tensor([0.1])

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
        mem1 = self.lif1.init_leaky()

        # Record the final layer
        spk_rec = []

        for step in range(x.size(1)):
            cur1 = self.fc1(x[:, step, :])
            spk1, mem1 = self.lif1(cur1, mem1)
            spk_rec.append(spk1)

        y = T.stack(spk_rec, dim=1)
        y = y.sum(dim=1)

        y = self.softmax(y)

        return y

    def mutate(self) -> 'LeakyBase':
        with T.no_grad():
            self.mutation_rate += T.normal(mean=0, std=T.Tensor([0.01]).to(self.dev.device))
            self.mutation_rate.data = T.clip(self.mutation_rate.data, min=0.001, max=1)
            self.mutation_std += T.normal(mean=0, std=T.Tensor([0.01]).to(self.dev.device))
            self.mutation_std.data = T.clip(self.mutation_std.data, min=0.001, max=3)
            self.mutation_std_0to1 += T.normal(mean=0, std=T.Tensor([0.1]).to(self.dev.device))
            self.mutation_std_0to1.data = T.clip(self.mutation_std_0to1.data, min=0.00001, max=1.0)

            mut_rate = self.mutation_rate[0].item()
            mut_std = self.mutation_std[0].item()

            for layer in self.layers:
                if isinstance(layer, nn.Linear):
                    layer.weight.data += generate_mutation_tensor(layer.weight.size(), mut_rate, mut_std, device=self.dev.device)
                    layer.bias.data += generate_mutation_tensor(layer.bias.size(), mut_rate, mut_std, device=self.dev.device)
                elif isinstance(layer, snn.Leaky):
                    layer.threshold.data += generate_mutation_tensor(layer.threshold.size(), mut_rate, mut_std, device=self.dev.device)
                    layer.threshold.data = T.clip(layer.threshold.data, min=0.0, max=10.0)

        return self

    def crossover(self, other: 'LeakyBase') -> 'LeakyBase':
        child = deepcopy(self).to(self.dev.device)
        child.load_state_dict(self.state_dict())

        with T.no_grad():
            child.mutation_rate.data = (self.mutation_rate.data + other.mutation_rate.data) / 2
            child.mutation_std.data = (self.mutation_std.data + other.mutation_std.data) / 2
            for layer_new, layer_p1, layer_p2 in zip(child.layers, self.layers, other.layers):
                if isinstance(layer_new, nn.Linear):
                    layer_new.weight.data = tensor_crossover(layer_p1.weight, layer_p2.weight)
                    layer_new.bias.data = tensor_crossover(layer_p1.bias, layer_p2.bias)
                elif isinstance(layer_new, snn.Leaky):
                    layer_new.threshold.data = (layer_p1.threshold.data + layer_p2.threshold.data) / 2

        return child


class LeakyOneLayer_NoBetaEvolution_Beta50pct(LeakyBase):
    def __init__(self):
        super().__init__()

        self.lif1.beta.data = T.Tensor([0.5])

        self.mutation_rate = nn.parameter.Parameter(T.Tensor([0.01]), requires_grad=False)
        self.mutation_std = nn.parameter.Parameter(T.Tensor([0.5]), requires_grad=False)

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
        mem1 = self.lif1.init_leaky()

        # Record the final layer
        spk_rec = []

        for step in range(x.size(1)):
            cur1 = self.fc1(x[:, step, :])
            spk1, mem1 = self.lif1(cur1, mem1)
            spk_rec.append(spk1)

        y = T.stack(spk_rec, dim=1)
        y = y.sum(dim=1)

        y = self.softmax(y)

        return y

    def mutate(self) -> 'LeakyBase':
        with T.no_grad():
            self.mutation_rate += T.normal(mean=0, std=T.Tensor([0.01]).to(self.dev.device))
            self.mutation_rate.data = T.clip(self.mutation_rate.data, min=0.001, max=1)
            self.mutation_std += T.normal(mean=0, std=T.Tensor([0.01]).to(self.dev.device))
            self.mutation_std.data = T.clip(self.mutation_std.data, min=0.001, max=3)
            self.mutation_std_0to1 += T.normal(mean=0, std=T.Tensor([0.1]).to(self.dev.device))
            self.mutation_std_0to1.data = T.clip(self.mutation_std_0to1.data, min=0.00001, max=1.0)

            mut_rate = self.mutation_rate[0].item()
            mut_std = self.mutation_std[0].item()

            for layer in self.layers:
                if isinstance(layer, nn.Linear):
                    layer.weight.data += generate_mutation_tensor(layer.weight.size(), mut_rate, mut_std, device=self.dev.device)
                    layer.bias.data += generate_mutation_tensor(layer.bias.size(), mut_rate, mut_std, device=self.dev.device)
                elif isinstance(layer, snn.Leaky):
                    layer.threshold.data += generate_mutation_tensor(layer.threshold.size(), mut_rate, mut_std, device=self.dev.device)
                    layer.threshold.data = T.clip(layer.threshold.data, min=0.0, max=10.0)

        return self

    def crossover(self, other: 'LeakyBase') -> 'LeakyBase':
        child = deepcopy(self).to(self.dev.device)
        child.load_state_dict(self.state_dict())

        with T.no_grad():
            child.mutation_rate.data = (self.mutation_rate.data + other.mutation_rate.data) / 2
            child.mutation_std.data = (self.mutation_std.data + other.mutation_std.data) / 2
            for layer_new, layer_p1, layer_p2 in zip(child.layers, self.layers, other.layers):
                if isinstance(layer_new, nn.Linear):
                    layer_new.weight.data = tensor_crossover(layer_p1.weight, layer_p2.weight)
                    layer_new.bias.data = tensor_crossover(layer_p1.bias, layer_p2.bias)
                elif isinstance(layer_new, snn.Leaky):
                    layer_new.threshold.data = (layer_p1.threshold.data + layer_p2.threshold.data) / 2

        return child


class LeakyOneLayer_NoBetaThresholdEvolution_Beta50pct_Threshold5(LeakyBase):
    def __init__(self):
        super().__init__()

        self.lif1.beta.data = T.Tensor([5])

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
        mem1 = self.lif1.init_leaky()

        # Record the final layer
        spk_rec = []

        for step in range(x.size(1)):
            cur1 = self.fc1(x[:, step, :])
            spk1, mem1 = self.lif1(cur1, mem1)
            spk_rec.append(spk1)

        y = T.stack(spk_rec, dim=1)
        y = y.sum(dim=1)

        y = self.softmax(y)

        return y

    def mutate(self) -> 'LeakyBase':
        with T.no_grad():
            self.mutation_rate += T.normal(mean=0, std=T.Tensor([0.01]).to(self.dev.device))
            self.mutation_rate.data = T.clip(self.mutation_rate.data, min=0.001, max=1)
            self.mutation_std += T.normal(mean=0, std=T.Tensor([0.01]).to(self.dev.device))
            self.mutation_std.data = T.clip(self.mutation_std.data, min=0.001, max=3)
            self.mutation_std_0to1 += T.normal(mean=0, std=T.Tensor([0.1]).to(self.dev.device))
            self.mutation_std_0to1.data = T.clip(self.mutation_std_0to1.data, min=0.00001, max=1.0)

            mut_rate = self.mutation_rate[0].item()
            mut_std = self.mutation_std[0].item()

            for layer in self.layers:
                if isinstance(layer, nn.Linear):
                    layer.weight.data += generate_mutation_tensor(layer.weight.size(), mut_rate, mut_std, device=self.dev.device)
                    layer.bias.data += generate_mutation_tensor(layer.bias.size(), mut_rate, mut_std, device=self.dev.device)
                elif isinstance(layer, snn.Leaky):
                    pass
                    # layer.threshold.data += generate_mutation_matrix(layer.threshold.size(), mut_rate, mut_std, device=self.dev.device)
                    # layer.threshold.data = T.clip(layer.threshold.data, min=0.0, max=10.0)

        return self

    def crossover(self, other: 'LeakyBase') -> 'LeakyBase':
        child = deepcopy(self).to(self.dev.device)
        child.load_state_dict(self.state_dict())

        with T.no_grad():
            child.mutation_rate.data = (self.mutation_rate.data + other.mutation_rate.data) / 2
            child.mutation_std.data = (self.mutation_std.data + other.mutation_std.data) / 2
            for layer_new, layer_p1, layer_p2 in zip(child.layers, self.layers, other.layers):
                if isinstance(layer_new, nn.Linear):
                    layer_new.weight.data = tensor_crossover(layer_p1.weight, layer_p2.weight)
                    layer_new.bias.data = tensor_crossover(layer_p1.bias, layer_p2.bias)
                elif isinstance(layer_new, snn.Leaky):
                    pass
                    # layer_new.threshold.data = (layer_p1.threshold.data + layer_p2.threshold.data) / 2

        return child


class LeakyOneLayer_NoBetaEvolution_HigherMutRateEvolution(LeakyBase):
    def __init__(self):
        super().__init__()

        self.lif1.beta.data = T.Tensor([0.1])

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
        mem1 = self.lif1.init_leaky()

        # Record the final layer
        spk_rec = []

        for step in range(x.size(1)):
            cur1 = self.fc1(x[:, step, :])
            spk1, mem1 = self.lif1(cur1, mem1)
            spk_rec.append(spk1)

        y = T.stack(spk_rec, dim=1)
        y = y.sum(dim=1)

        y = self.softmax(y)

        return y

    def mutate(self) -> 'LeakyBase':
        with T.no_grad():
            self.mutation_rate += T.normal(mean=0, std=T.Tensor([0.1]).to(self.dev.device))
            self.mutation_rate.data = T.clip(self.mutation_rate.data, min=0.001, max=1)
            self.mutation_std += T.normal(mean=0, std=T.Tensor([0.01]).to(self.dev.device))
            self.mutation_std.data = T.clip(self.mutation_std.data, min=0.001, max=3)
            self.mutation_std_0to1 += T.normal(mean=0, std=T.Tensor([0.1]).to(self.dev.device))
            self.mutation_std_0to1.data = T.clip(self.mutation_std_0to1.data, min=0.00001, max=1.0)

            mut_rate = self.mutation_rate[0].item()
            mut_std = self.mutation_std[0].item()

            for layer in self.layers:
                if isinstance(layer, nn.Linear):
                    layer.weight.data += generate_mutation_tensor(layer.weight.size(), mut_rate, mut_std, device=self.dev.device)
                    layer.bias.data += generate_mutation_tensor(layer.bias.size(), mut_rate, mut_std, device=self.dev.device)
                elif isinstance(layer, snn.Leaky):
                    layer.threshold.data += generate_mutation_tensor(layer.threshold.size(), mut_rate, mut_std, device=self.dev.device)
                    layer.threshold.data = T.clip(layer.threshold.data, min=0.0, max=10.0)

        return self

    def crossover(self, other: 'LeakyBase') -> 'LeakyBase':
        child = deepcopy(self).to(self.dev.device)
        child.load_state_dict(self.state_dict())

        with T.no_grad():
            child.mutation_rate.data = (self.mutation_rate.data + other.mutation_rate.data) / 2
            child.mutation_std.data = (self.mutation_std.data + other.mutation_std.data) / 2
            for layer_new, layer_p1, layer_p2 in zip(child.layers, self.layers, other.layers):
                if isinstance(layer_new, nn.Linear):
                    layer_new.weight.data = tensor_crossover(layer_p1.weight, layer_p2.weight)
                    layer_new.bias.data = tensor_crossover(layer_p1.bias, layer_p2.bias)
                elif isinstance(layer_new, snn.Leaky):
                    layer_new.threshold.data = (layer_p1.threshold.data + layer_p2.threshold.data) / 2

        return child


class LeakyTwoLayer_NoBetaEvolution_Beta50pct(LeakyBase):
    def __init__(self):
        super().__init__()

        self.lif1.beta.data = T.Tensor([0.5])

        self.fc1 = nn.Linear(self.num_inputs, 50)
        self.lif1 = snn.Leaky(beta=0.5, threshold=1.0)
        self.fc2 = nn.Linear(50, self.num_outputs)
        self.lif2 = snn.Leaky(beta=0.5, threshold=1.0)

        self.mutation_rate = nn.parameter.Parameter(T.Tensor([0.01]), requires_grad=False)
        self.mutation_std = nn.parameter.Parameter(T.Tensor([0.5]), requires_grad=False)

        self.layers = [
            self.fc1,
            self.lif1,
            self.fc2,
            self.lif2,
        ]

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
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # Record the final layer
        spk_rec = []

        for step in range(x.size(1)):
            cur1 = self.fc1(x[:, step, :])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk_rec.append(spk2)

        y = T.stack(spk_rec, dim=1)
        y = y.sum(dim=1)

        y = self.softmax(y)

        return y

    def mutate(self) -> 'LeakyBase':
        with T.no_grad():
            self.mutation_rate += T.normal(mean=0, std=T.Tensor([0.01]).to(self.dev.device))
            self.mutation_rate.data = T.clip(self.mutation_rate.data, min=0.001, max=1)
            self.mutation_std += T.normal(mean=0, std=T.Tensor([0.01]).to(self.dev.device))
            self.mutation_std.data = T.clip(self.mutation_std.data, min=0.001, max=3)
            self.mutation_std_0to1 += T.normal(mean=0, std=T.Tensor([0.1]).to(self.dev.device))
            self.mutation_std_0to1.data = T.clip(self.mutation_std_0to1.data, min=0.00001, max=1.0)

            mut_rate = self.mutation_rate[0].item()
            mut_std = self.mutation_std[0].item()

            for layer in self.layers:
                if isinstance(layer, nn.Linear):
                    layer.weight.data += generate_mutation_tensor(layer.weight.size(), mut_rate, mut_std, device=self.dev.device)
                    layer.bias.data += generate_mutation_tensor(layer.bias.size(), mut_rate, mut_std, device=self.dev.device)
                elif isinstance(layer, snn.Leaky):
                    layer.threshold.data += generate_mutation_tensor(layer.threshold.size(), mut_rate, mut_std, device=self.dev.device)
                    layer.threshold.data = T.clip(layer.threshold.data, min=0.0, max=10.0)

        return self

    def crossover(self, other: 'LeakyBase') -> 'LeakyBase':
        child = deepcopy(self).to(self.dev.device)
        child.load_state_dict(self.state_dict())

        with T.no_grad():
            child.mutation_rate.data = (self.mutation_rate.data + other.mutation_rate.data) / 2
            child.mutation_std.data = (self.mutation_std.data + other.mutation_std.data) / 2
            for layer_new, layer_p1, layer_p2 in zip(child.layers, self.layers, other.layers):
                if isinstance(layer_new, nn.Linear):
                    layer_new.weight.data = tensor_crossover(layer_p1.weight, layer_p2.weight)
                    layer_new.bias.data = tensor_crossover(layer_p1.bias, layer_p2.bias)
                elif isinstance(layer_new, snn.Leaky):
                    layer_new.threshold.data = (layer_p1.threshold.data + layer_p2.threshold.data) / 2

        return child

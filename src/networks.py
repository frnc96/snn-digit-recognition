import snntorch as snn
import torch as T
import torch.nn as nn


def generate_mutation_matrix(size, prob, std, device):
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

# Define Network


class Net(nn.Module):
    def __init__(
        self,
        num_inputs: int = 28*28,
        num_hidden: int = 10,
        num_outputs: int = 10,
        beta: float = 0.9,
        threshold: float = 1.0,
        mutation_rate: float = 0.1,
        mutation_std: float = 0.5,

    ):
        super().__init__()

        self.leaky_beta = beta
        self.leaky_threshold = threshold
        self.softmax = T.nn.Softmax(dim=1)

        # Initialize SNN layers
        self.fc1 = nn.Linear(num_inputs, num_outputs)
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold)

        # self.fc2 = nn.Linear(num_hidden, num_outputs)
        # self.lif2 = snn.Leaky(beta=beta, threshold=threshold)

        # self.fc3 = nn.Linear(num_hidden, num_outputs)
        # self.lif3 = snn.Leaky(beta=beta, threshold=threshold)

        self.dev = T.nn.parameter.Parameter(T.empty([1]), requires_grad=False)
        self.mutation_rate = nn.parameter.Parameter(T.Tensor([mutation_rate]), requires_grad=False)
        self.mutation_std = nn.parameter.Parameter(T.Tensor([mutation_std]), requires_grad=False)

        self.layers = [
            self.fc1,
            self.lif1,
            # self.fc2,
            # self.lif2,
            # self.fc3,
            # self.lif3,
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
        # mem2 = self.lif2.init_leaky()
        # mem3 = self.lif3.init_leaky()

        # Record the final layer
        spk_rec = []

        for step in range(x.size(1)):
            cur1 = self.fc1(x[:, step, :])
            spk1, mem1 = self.lif1(cur1, mem1)
            # cur2 = self.fc2(spk1)
            # spk2, mem2 = self.lif2(cur2, mem2)
            # cur3 = self.fc3(spk2)
            # spk3, mem3 = self.lif3(cur3, mem3)
            spk_rec.append(spk1)

        y = T.stack(spk_rec, dim=1)
        y = y.sum(dim=1)

        y = self.softmax(y)

        return y

    def mutate(self):
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
                elif isinstance(layer, snn.Leaky):
                    layer.beta.data += generate_mutation_matrix(layer.beta.size(), mut_rate, mut_std, device=self.dev.device)
                    layer.threshold.data += generate_mutation_matrix(layer.threshold.size(), mut_rate, mut_std, device=self.dev.device)

        return self

    def crossover(self, other: 'Net'):
        child = Net().to(self.dev.device)
        child.load_state_dict(self.state_dict())

        with T.no_grad():
            child.mutation_rate.data = (self.mutation_rate.data + other.mutation_rate.data) / 2
            child.mutation_std.data = (self.mutation_std.data + other.mutation_std.data) / 2
            for layer_new, layer_p1, layer_p2 in zip(child.layers, self.layers, other.layers):
                if isinstance(layer_new, nn.Linear):
                    layer_new.weight.data = tensor_crossover(layer_p1.weight, layer_p2.weight)
                    layer_new.bias.data = tensor_crossover(layer_p1.bias, layer_p2.bias)
                elif isinstance(layer_new, snn.Leaky):
                    layer_new.beta.data = tensor_crossover(layer_p1.beta, layer_p2.beta)
                    layer_new.threshold.data = tensor_crossover(layer_p1.threshold, layer_p2.threshold)

        return child

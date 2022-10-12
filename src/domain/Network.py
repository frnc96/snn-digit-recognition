import torch
import torch.nn as nn
import snntorch as snn
import src.domain.constants.parameters as params


# Define Network
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # Initialize layers
        self.fc1 = nn.Linear(params.NUM_OF_INPUT_NEURONS, params.NUM_OF_HIDDEN_NEURONS)
        self.lif1 = snn.Leaky(beta=params.LEAKY_BETA)
        self.fc2 = nn.Linear(params.NUM_OF_HIDDEN_NEURONS, params.NUM_OF_OUTPUT_NEURONS)
        self.lif2 = snn.Leaky(beta=params.LEAKY_BETA)

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

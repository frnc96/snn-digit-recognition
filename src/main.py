from frencis.Data import Encoder
from frencis.Network import Net
from frencis.Training import BackpropTT
import torch

# Set the device to run on GPU if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Init network
network = Net().to(device)

# Init data encoder
encoder = Encoder()

# Init trainer
backprop_tt = BackpropTT(network)

train_loader = encoder.get_train_loader()
test_loader = encoder.get_test_loader()

# Run the training loop
backprop_tt.training_loop(train_loader=train_loader, test_loader=test_loader)

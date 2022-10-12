from domain.Data import Encoder
from domain.Network import Net
from domain.Training import BackpropTT
import torch

# Set the device to run on GPU if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Init network
network = Net().to(device)

# Init data encoder
encoder = Encoder()

# Init trainer
backprop_tt = BackpropTT(net=network)

# Get the training and testing data loaders
train_loader, test_loader = encoder.get_loaders()

# Run the training loop
backprop_tt.train(train_loader, test_loader)

# Test the model
backprop_tt.test(test_loader)

# Save the trained model
torch.save(network.state_dict(), "../models/1-epoch-snn-backprop-cpu")
print("Model saved...")

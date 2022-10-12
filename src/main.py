from frencis.Data import Encoder
from frencis.Network import Net
from frencis.Training import BackpropTT
import torch

# Parameters START
num_inputs = 28 * 28    # Number of input neurons
num_hidden = 1000       # Number of neurons in the hidden layers
num_outputs = 10        # Number of output neurons
num_steps = 25          # Number of steps
beta = 0.95             # Leaky factor of the neurons
batch_size = 128        # Dataset batch size
num_epochs = 1          # Number of epochs to train for
# Parameters END

# Set the device to run on GPU if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Init network
network = Net(num_inputs, num_hidden, num_outputs, num_steps, beta).to(device)

# Init data encoder
encoder = Encoder(batch_size)

# Init trainer
backprop_tt = BackpropTT(network, num_epochs, batch_size, num_steps)

# Get the training and testing data loaders
train_loader, test_loader = encoder.get_loaders()

# Run the training loop
backprop_tt.train(train_loader, test_loader)

# Test the model
backprop_tt.test(test_loader)

# Save the trained model
torch.save(network.state_dict(), "models/1-epoch-snn-backprop-cpu")
print("Model saved...")

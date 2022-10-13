import torch

# Set the device to run on GPU if available
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

TRAINING_TYPE = "EVO"               # Should we train using backprop or a genetic algorithm (EVO, BTT)

NUM_OF_INPUT_NEURONS = 28 * 28      # Number of input neurons
NUM_OF_HIDDEN_NEURONS = 1000        # Number of neurons in the hidden layers
NUM_OF_OUTPUT_NEURONS = 10          # Number of output neurons
LEAKY_BETA = 0.95                   # Leaky factor of the neurons

NUM_OF_EPOCHS = 1                   # Number of epochs to train for
NUM_OF_STEPS = 25                   # Number of steps
BATCH_SIZE = 128                    # Dataset batch size

POPULATION_SIZE = 50                # Number of agents in a generation
NUM_OF_GENERATIONS = 3              # For how many generations should we train

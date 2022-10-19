import torch

# Set the device to run on GPU if available
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

TRAINING_TYPE = "EVO"               # Should we train using backprop or a genetic algorithm (EVO, BTT)

# Spiking neural network parameters
NUM_OF_INPUT_NEURONS = 28 * 28      # Number of input neurons
NUM_OF_HIDDEN_NEURONS = 20          # Number of neurons in the hidden layers
NUM_OF_OUTPUT_NEURONS = 10          # Number of output neurons
LEAKY_BETA = 0.95                   # Leaky factor of the neurons

# Backprop algorithm parameters
NUM_OF_EPOCHS = 10                  # Number of epochs to train for
NUM_OF_STEPS = 25                   # Number of steps
BATCH_SIZE = 128                    # Dataset batch size

# Genetic algorithm parameters
NUM_OF_GENERATIONS = 100            # For how many generations should we train
POPULATION_SIZE = 4                 # Number of agents in a generation (must be at least 4)
CHILDREN_RATIO = 0.5                # Percentage of population to be replaced by children
MUTATION_RATE = 0.001               # Agent mutation rate
MUTATION_STD = 0.001                # Agent mutation standard deviation

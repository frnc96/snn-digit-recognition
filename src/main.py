from domain.Data import Encoder
from domain.Network import Net
from domain.Training import BackpropTT
from domain.Evolution import GeneticAlgorithm
import src.domain.constants.parameters as params
import torch

# Init data encoder
encoder = Encoder()
train_loader, test_loader = encoder.get_loaders()

if params.TRAINING_TYPE == "BTT":
    print("Training a single SNN using backprop through time")

    # Init network
    backpropModel = Net().to(params.DEVICE)

    # Train a single network using backpropTT
    backprop_tt = BackpropTT(net=backpropModel)
    backprop_tt.train(train_loader, test_loader)
    backprop_tt.test(test_loader)

    # Save the trained backprop model
    torch.save(backpropModel.state_dict(), f"../models/{params.NUM_OF_EPOCHS}-epoch-snn-backprop-{params.DEVICE}.pth")
    print("Backprop model saved...")

elif params.TRAINING_TYPE == "EVO":
    print("Training a population of networks using a genetic algorithm")

    # Train the networks with genetic algorithm
    evo = GeneticAlgorithm(train_loader, test_loader)
    geneticModel = evo.train().get_global_best()

    # Save the trained evolutionary model
    torch.save(geneticModel.state_dict(), f"../models/{params.NUM_OF_GENERATIONS}-gen-snn-evolutionary-{params.DEVICE}.pth")
    print("Evolutionary model saved...")

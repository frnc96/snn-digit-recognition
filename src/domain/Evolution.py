import torch
import matplotlib.pyplot as plt
from src.domain.Network import Net
from src.domain.utilities.helpers import EvoHelpers
import src.domain.constants.parameters as params


def plot(y_values):
    x_values = list(range(0, params.NUM_OF_GENERATIONS + 1))

    plt.xlabel("Generation")
    plt.ylabel("Loss")

    plt.plot(x_values, y_values)
    plt.show()


class GeneticAlgorithm:

    def __init__(self, train_loader, test_loader):
        self.train_loader = train_loader
        self.test_loader = test_loader

        # Initialize the population of networks
        self.population = []
        for i in range(params.POPULATION_SIZE):
            # Init the loss as None
            self.population.append([None, Net().to(params.DEVICE)])

        print(f"Initialized a population with {params.POPULATION_SIZE} members")
        self.sort_population()

    def sort_population(self):
        print("Evaluating loss for each agent and sorting them...")

        counter = 1
        population = []
        for loss, agent in self.population:
            # Calculate loss only if new agent
            if loss is None:
                loss = EvoHelpers.evaluate_loss(agent, self.train_loader, counter)
            population.append((loss, agent))
            counter += 1

        # Sort the population by loss
        self.population = sorted(population, key=lambda x: x[0])

        return self

    def train(self):
        y_values = [0] * (params.NUM_OF_GENERATIONS + 1)
        y_values[0] = self.get_mean_loss()

        # Get population slice index
        slice_best = int(len(self.population) * params.CHILDREN_RATIO)

        # Outer training loop
        for generation_number in range(1, params.NUM_OF_GENERATIONS + 1):
            print(f"Generation {generation_number}/{params.NUM_OF_GENERATIONS}")

            # Keep only the first 50%
            self.population = self.population[:slice_best]

            # The fittest 50% will be the parents
            parents = self.population

            # Init the children list
            children = []

            # Crossover the parents and mutate the children
            for agent_index in range(0, len(parents), 2):
                if len(parents) + 1 == agent_index:
                    continue

                parent_one: Net = parents[agent_index][-1]
                parent_two: Net = parents[agent_index + 1][-1]

                child_one: Net = parent_one.crossover(parent_two).mutate()
                child_two: Net = parent_one.crossover(parent_two).mutate()
                children.append((None, child_one))
                children.append((None, child_two))

            # Add children to the population
            self.population += children

            # Set loss for new children and sort
            self.sort_population()

            # Update plot
            y_values[generation_number] = self.get_mean_loss()
            plot(y_values)

        return self

    def evaluate_accuracy(self, net: Net):
        total = 0
        correct = 0

        # Iterate through all mini-batches and measure accuracy over the full data set
        with torch.no_grad():
            net.eval()
            for data, targets in self.test_loader:
                data = data.to(params.DEVICE)
                targets = targets.to(params.DEVICE)

                # forward pass
                test_spk, _ = net(data.view(data.size(0), -1))

                # calculate total accuracy
                _, predicted = test_spk.sum(dim=0).max(1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        return correct / total

    def get_mean_loss(self):
        loss_list = list(map(lambda x: x[0], self.population))
        return sum(loss_list) / len(loss_list)

    def get_global_best(self):
        loss = self.population[0][0]
        agent = self.population[0][-1]
        accuracy = self.evaluate_accuracy(agent)

        print(f"Best agent loss = {loss}")
        print(f"Best agent acc = {100 * accuracy:.2f}%")

        # Return the fittest
        return agent

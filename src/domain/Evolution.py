import torch
import random
import matplotlib.pyplot as plt
from src.domain.Network import Net
from src.domain.utilities.helpers import EvoHelpers
import src.domain.constants.parameters as params


def plot(x_values, y_values):
    plt.xlabel("Generation")
    plt.ylabel("Loss")

    plt.xlim((1, params.NUM_OF_GENERATIONS))
    plt.ylim((0, 3))

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
        x_values = []
        y_values = []

        # Get population slice index
        slice_best = int(len(self.population) * params.CHILDREN_RATIO)

        # Outer training loop
        for generation_number in range(1, params.NUM_OF_GENERATIONS + 1):
            print(f"Generation {generation_number}/{params.NUM_OF_GENERATIONS}")

            parents = self.population

            # Keep only the first 50%
            self.population = self.population[:slice_best]

            # Keep generating children from randomly selected parents
            while len(self.population) < params.POPULATION_SIZE:
                parent_one: Net = self.get_random_parent(parents)
                parent_two: Net = self.get_random_parent(parents)

                child: Net = parent_one.crossover(parent_two).mutate()

                # Append the child to the population
                self.population.append((None, child))

            # Set loss for new children and sort
            self.sort_population()
            self.get_global_best()

            # Update plot
            x_values.append(generation_number)
            y_values.append(self.get_mean_loss())
            plot(x_values, y_values)

        return self

    def evaluate_accuracy(self, net: Net) -> float:
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

    @staticmethod
    def get_random_parent(parents: list) -> Net:
        # todo - randomly select based in probability
        return random.choice(parents)[-1]

    def get_mean_loss(self):
        loss_list = list(map(lambda x: x[0], self.population))
        return sum(loss_list) / len(loss_list)

    def get_global_best(self):
        loss = self.population[0][0]
        agent = self.population[0][-1]
        accuracy = 100 * self.evaluate_accuracy(agent)

        print(f"Best agent loss = {loss}")
        print(f"Best agent acc = {accuracy:.2f}%")

        # Return the fittest
        return agent, round(accuracy)

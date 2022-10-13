from src.domain.Network import Net
from src.domain.utilities.helpers import EvoHelpers
import src.domain.constants.parameters as params


class GeneticAlgorithm:

    def __init__(self, train_loader, test_loader):
        self.train_loader = train_loader
        self.test_loader = test_loader

        # Initialize the population of networks
        self.population = []
        for i in range(params.POPULATION_SIZE):
            # Init the loss as 0
            self.population.append([0, Net().to(params.DEVICE)])

        print(f"Initialized a population with {params.POPULATION_SIZE} members")

    def sort_population(self):
        self.population = sorted([
            (
                # todo - fix train_loader
                EvoHelpers.evaluate_loss(
                    agent[-1],
                    self.train_loader,
                    self.train_loader,
                    with_replacement=True,
                    n_samples=100
                ),
                agent[-1]
            )
            for agent in self.population
        ], key=lambda x: x[0])

        return self

    def train(self):
        slice_best = int(len(self.population) * 0.8)
        slice_rest = int(len(self.population) * 0.4)

        # Outer training loop
        for generation_number in range(params.NUM_OF_GENERATIONS):
            print(f"Generation {generation_number}")

            # Sort population by fittness (loss)
            self.sort_population()

            # Keep only the first 80%
            self.population = self.population[:slice_best]

            # The fittest 40% will be the parents
            parents = self.population[:slice_rest]

            # Init the children list
            children = []

            # Crossover the parents and mutate the children
            for agent_index in range(0, len(parents), 2):
                parent_one: Net = parents[agent_index][-1]
                parent_two: Net = parents[agent_index + 1][-1]

                child: Net = parent_one.crossover(parent_two).mutate()

                children.append([9999, child])

            # Add children to the population
            self.population += children

        return self

    def get_global_best(self):
        # Sort population once more
        self.sort_population()

        # Return the fittest
        if self.population:
            return self.population[0][-1]

        # Or None if population is empty
        return None

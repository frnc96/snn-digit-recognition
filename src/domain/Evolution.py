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
            # Init the loss as None
            self.population.append([None, Net().to(params.DEVICE)])

        print(f"Initialized a population with {params.POPULATION_SIZE} members")

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
        slice_best = int(len(self.population) * params.CHILDREN_RATIO)

        # Outer training loop
        for generation_number in range(params.NUM_OF_GENERATIONS):
            print(f"Generation {generation_number}/{params.NUM_OF_GENERATIONS}")

            # Sort population by fittness (loss)
            self.sort_population()

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

        return self

    def get_global_best(self):
        # Sort population once more
        self.sort_population()

        # Return the fittest
        if self.population:
            print(f"Best agent loss = {self.population[0][0]}")
            return self.population[0][-1]

        # Or None if population is empty
        return None

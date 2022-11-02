from abc import ABC, abstractmethod


class BaseNetwork(ABC):
    """Base class for our neural networks."""

    @abstractmethod
    def describe(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def mutate(self) -> 'BaseNetwork':
        raise NotImplementedError

    @abstractmethod
    def crossover(self, other: 'BaseNetwork') -> 'BaseNetwork':
        raise NotImplementedError

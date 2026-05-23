from abc import ABC, abstractmethod

class TransformBase(ABC):
    @abstractmethod
    def __call__(
        self,
        sample
    ):
        pass
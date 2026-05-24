from abc import ABC, abstractmethod
from typing import Dict

class TransformBase(ABC):
    @abstractmethod
    def __call__(
        self,
        sample: Dict
    ) -> Dict:
        pass
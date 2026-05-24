from abc import ABC, abstractmethod
from typing import Dict, Any, List


class TaskSpec(ABC):
    """
    Base contract for all tasks in the framework.
    """

    name: str

    # keys expected from dataset batch
    input_keys: List[str]
    target_keys: List[str]

    # keys expected from model output
    output_keys: List[str]

    # loss and metrics definitions (configs or objects)
    losses: List[Any]
    metrics: List[Any]

    def __init__(self):
        pass

    @abstractmethod
    def format_targets(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Converts raw dataset batch into task-ready targets.
        """
        pass

    @abstractmethod
    def format_outputs(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensures model outputs match expected structure.
        """
        pass
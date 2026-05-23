from abc import ABC, abstractmethod
from torch.utils.data import Dataset

class DatasetBase(Dataset, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass




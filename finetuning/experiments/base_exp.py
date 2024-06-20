from abc import ABCMeta, abstractmethod
from typing import Dict
import torch
from torch.nn import Module

class BaseClass(metaclass=ABCMeta):
    """     Basic class for any experiment.     """

    def __init__(self):
        self.seed = None
        self.output_dir = "./outputs"
        self.print_interval = 100
        self.eval_interval = 10
        self.dataset = None

    @abstractmethod
    def get_model(self) -> Module:
        pass

    @abstractmethod
    def get_train_dataset(self):
        pass

    @abstractmethod
    def get_train_loader(
        self, batch_size: int
    ) -> Dict[str, torch.utils.data.DataLoader]:
        pass

    # @abstractmethod
    # def eval(self, model, evaluator, weights):
    #     pass
from argparse import Namespace
from typing import Optional
from abc import ABC, abstractmethod

from addict import Dict as Adict
from fas_simple_distill.utils.general import seed_all


class BaseTrainer(ABC):
    def __init__(
        self, args: Namespace, config: Adict, checkpoint_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.args = args
        self.config = Adict(config)
        self.config.freeze(True)

        seed_all(self.config.hparams.seed)

        if self.config.use_cpu:
            self.device = "cpu"
        else:
            self.device = "cuda"

        self.start_epoch: int = 0
        self.current_epoch: int = self.start_epoch
        self.global_step: int = 0
        self.iter_per_epoch: int = self.config.hparams.iter_per_epoch
        self.max_iter: int = self.config.hparams.max_iter

        self.build_model()
        self.configure_optimizers()
        self.initialize_train_loader()
        self.initialize_val_loaders()

        if self.config.hparams.get("pretrained_path", False):
            self.load_checkpoint(self.config.hparams.pretrained_path)
        
    @abstractmethod
    def initialize_train_loader(self):
        raise NotImplementedError
    
    @abstractmethod
    def initialize_val_loaders(self):
        raise NotImplementedError
    
    @abstractmethod
    def build_model(self):
        raise NotImplementedError
    
    @abstractmethod
    def configure_optimizers(self):
        raise NotImplementedError
    
    @abstractmethod
    def load_checkpoint(self, checkpoint_dir):
        raise NotImplementedError


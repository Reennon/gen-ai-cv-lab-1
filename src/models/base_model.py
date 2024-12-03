import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger


class BaseModel(pl.LightningModule):
    def __init__(self, hparams):
        super(BaseModel, self).__init__()
        self.save_hyperparameters(hparams)
        # Define your model architecture here
        # For now, we'll leave it abstract

    def forward(self, x):
        # Define the forward pass
        pass

    def training_step(self, batch, batch_idx):
        # Implement the training logic
        pass

    def validation_step(self, batch, batch_idx):
        # Implement the validation logic
        pass

    def configure_optimizers(self):
        optimizer_config = self.hparams['optimizer']
        optimizer_type = optimizer_config['type']
        optimizer_params = {k: v for k, v in optimizer_config.items() if k != 'type'}
        optimizer = getattr(optim, optimizer_type)(self.parameters(), **optimizer_params)

        scheduler_config = self.hparams.get('scheduler', None)
        if scheduler_config and scheduler_config['type'] is not None:
            scheduler_type = scheduler_config['type']
            scheduler_params = scheduler_config['params']
            scheduler = getattr(self.hparams.lr, scheduler_type)(optimizer, **scheduler_params)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',  # or 'step'
                    'monitor': 'val_loss'  # for schedulers like ReduceLROnPlateau
                }
            }
        else:
            return optimizer

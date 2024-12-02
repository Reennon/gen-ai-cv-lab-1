import torch
import pytorch_lightning as pl


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
        # Set up the optimizer and learning rate scheduler
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams['lr'])
        return optimizer

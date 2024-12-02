import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import torch


def train_model(model_class, hparams, train_loader, val_loader):
    """
    Trains a model using PyTorch Lightning.

    Args:
        model_class: The model class to be instantiated and trained.
        hparams: A dictionary of hyperparameters for training.
        train_loader: DataLoader for the training dataset.
        val_loader: DataLoader for the validation dataset.
    """
    # Initialize the model
    model = model_class(hparams)

    # Initialize Wandb logger
    wandb_logger = WandbLogger(
        project='CIFAR10-Training',
        name=hparams.get('run_name', 'Experiment'),
        log_model=True
    )

    # Initialize PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=hparams['epochs'],
        logger=wandb_logger,
        gpus=1 if torch.cuda.is_available() else 0,
        callbacks=[
            ModelCheckpoint(
                monitor='val_loss',
                save_top_k=1,
                mode='min'
            )
        ]
    )

    # Start training
    trainer.fit(model, train_loader, val_loader)

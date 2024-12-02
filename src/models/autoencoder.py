import torch
import torch.nn as nn
import torchvision
import wandb

from pytorch_lightning.loggers import WandbLogger
from src.models.base_model import BaseModel


class Autoencoder(BaseModel):
    def __init__(self, hparams):
        super(Autoencoder, self).__init__()
        self.save_hyperparameters(hparams)
        # Define the encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # Output: (64, 16, 16)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # Output: (128, 8, 8)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # Output: (256, 4, 4)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, self.hparams['latent_dim'])  # Latent representation
        )

        # Define the decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.hparams['latent_dim'], 256 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (256, 4, 4)),  # Reshape to match the encoder's output
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # Output: (128, 8, 8)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # Output: (64, 16, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),  # Output: (3, 32, 32)
            nn.Sigmoid()  # Scale output to [0, 1]
        )

        # Initialize a list to store validation outputs
        self.validation_outputs = []

    def forward(self, x):
        # Pass through encoder
        z = self.encoder(x)
        # Pass through decoder
        x_hat = self.decoder(z)
        return x_hat

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self.forward(x)
        loss = nn.functional.mse_loss(x_hat, x)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self.forward(x)
        val_loss = nn.functional.mse_loss(x_hat, x)
        self.log('val_loss', val_loss)
        # Store outputs for later use
        self.validation_outputs.append((x, x_hat))
        return val_loss

    def on_validation_epoch_end(self):
        """
        Hook to log reconstructed images at the end of the validation epoch.
        Logs the original and reconstructed images as a grid to Weights & Biases.
        """
        # Ensure the logger is an instance of WandbLogger
        if isinstance(self.logger, WandbLogger):
            # Select a few samples from the validation outputs
            outputs = self.validation_outputs[:8]  # Take the first 8 batches
            original_images, reconstructed_images = zip(*outputs)

            # Combine images into a single tensor
            original_images = torch.cat(original_images, dim=0)
            reconstructed_images = torch.cat(reconstructed_images, dim=0)

            # Create grids of original and reconstructed images
            original_grid = torchvision.utils.make_grid(original_images.cpu(), nrow=4, normalize=True)
            reconstructed_grid = torchvision.utils.make_grid(reconstructed_images.cpu(), nrow=4, normalize=True)

            # Log the images to W&B
            self.logger.experiment.log({
                "Original Images": [wandb.Image(original_grid, caption='Original Images')],
                "Reconstructed Images": [wandb.Image(reconstructed_grid, caption='Reconstructed Images')]
            })

        # Clear the stored validation outputs
        self.validation_outputs.clear()

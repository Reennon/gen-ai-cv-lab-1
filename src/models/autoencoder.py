import torch

from src.models.base_model import BaseModel


class Autoencoder(BaseModel):
    def __init__(self, hparams):
        super(Autoencoder, self).__init__(hparams)
        # Define encoder
        self.encoder = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(32*32*3, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, hparams['latent_dim'])
        )
        # Define decoder
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(hparams['latent_dim'], 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 32*32*3),
            torch.nn.Sigmoid(),
            torch.nn.Unflatten(1, (3, 32, 32))
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self.forward(x)
        loss = torch.nn.functional.mse_loss(x_hat, x)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self.forward(x)
        val_loss = torch.nn.functional.mse_loss(x_hat, x)
        self.log('val_loss', val_loss)
        return val_loss

    def validation_epoch_end(self, outputs):
        # Log reconstructed images
        sample_imgs = next(iter(val_loader))[0][:8]  # Get a batch of images
        reconstructions = self.forward(sample_imgs.to(self.device))
        grid = torchvision.utils.make_grid(reconstructions.cpu())
        self.logger.experiment.log({'reconstructions': [wandb.Image(grid, caption='Reconstructed Images')]})


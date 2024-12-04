import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import wandb

from src.models.base_model import BaseModel


class GAN(BaseModel):
    def __init__(self, hparams):
        super(GAN, self).__init__(hparams)
        self.save_hyperparameters(hparams)

        # Generator
        self.generator = nn.Sequential(
            nn.Linear(self.hparams['latent_dim'], 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 3 * 32 * 32),  # Assuming output is CIFAR-10 images
            nn.Tanh(),  # Scale output to [-1, 1]
        )

        # Discriminator
        self.discriminator = nn.Sequential(
            nn.Linear(3 * 32 * 32, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),  # Output a probability
        )

        self.criterion = nn.BCELoss()

    def forward(self, z):
        """Generate images from latent space."""
        return self.generator(z)

    def training_step(self, batch, batch_idx, optimizer_idx):
        real_imgs, _ = batch
        real_imgs = real_imgs.view(real_imgs.size(0), -1)  # Flatten images

        # Labels
        real_labels = torch.ones(real_imgs.size(0), 1, device=self.device)
        fake_labels = torch.zeros(real_imgs.size(0), 1, device=self.device)

        # Generator optimization
        if optimizer_idx == 0:
            z = torch.randn(real_imgs.size(0), self.hparams['latent_dim'], device=self.device)
            fake_imgs = self(z)
            g_loss = self.criterion(self.discriminator(fake_imgs), real_labels)
            self.log('g_loss', g_loss)
            return g_loss

        # Discriminator optimization
        if optimizer_idx == 1:
            # Discriminator loss on real images
            real_loss = self.criterion(self.discriminator(real_imgs), real_labels)

            # Discriminator loss on fake images
            z = torch.randn(real_imgs.size(0), self.hparams['latent_dim'], device=self.device)
            fake_imgs = self(z).detach()
            fake_loss = self.criterion(self.discriminator(fake_imgs), fake_labels)

            d_loss = (real_loss + fake_loss) / 2
            self.log('d_loss', d_loss)
            return d_loss

    def configure_optimizers(self):
        g_optimizer = optim.Adam(self.generator.parameters(), lr=self.hparams['lr'], betas=(0.5, 0.999))
        d_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.hparams['lr'], betas=(0.5, 0.999))
        return [g_optimizer, d_optimizer], []

    def on_validation_epoch_end(self):
        """Log generated images at the end of each validation epoch."""
        z = torch.randn(8, self.hparams['latent_dim'], device=self.device)
        fake_imgs = self(z).view(-1, 3, 32, 32)  # Reshape for grid
        grid = torchvision.utils.make_grid(fake_imgs, nrow=4, normalize=True)
        self.logger.experiment.log({
            "Generated Images": [wandb.Image(grid, caption='Generated Images')]
        })

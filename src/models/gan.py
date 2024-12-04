import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim
from src.models.base_model import BaseModel


class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 3 * 32 * 32),  # CIFAR-10 output dimensions
            nn.Tanh()  # Scale output to [-1, 1]
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), 3, 32, 32)  # Reshape to image dimensions
        return img



class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3 * 32 * 32, 512),  # Flatten CIFAR-10 input
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Output probability
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)  # Flatten the input image
        validity = self.model(img_flat)
        return validity



class GAN(pl.LightningModule):
    def __init__(self, latent_dim, lr=0.0002):
        super(GAN, self).__init__()
        self.latent_dim = latent_dim
        self.generator = Generator(latent_dim)
        self.discriminator = Discriminator()
        self.lr = lr

    def forward(self, z):
        return self.generator(z)

    def training_step(self, batch, batch_idx, optimizer_idx):
        real_imgs, _ = batch

        # Sample random noise for the generator
        batch_size = real_imgs.size(0)
        z = torch.randn(batch_size, self.latent_dim, device=self.device)

        # Generator step
        if optimizer_idx == 0:
            fake_imgs = self(z)
            fake_preds = self.discriminator(fake_imgs)
            g_loss = nn.BCELoss()(fake_preds, torch.ones_like(fake_preds))
            self.log('g_loss', g_loss, prog_bar=True)
            return g_loss

        # Discriminator step
        if optimizer_idx == 1:
            real_preds = self.discriminator(real_imgs)
            real_loss = nn.BCELoss()(real_preds, torch.ones_like(real_preds))

            fake_imgs = self(z).detach()
            fake_preds = self.discriminator(fake_imgs)
            fake_loss = nn.BCELoss()(fake_preds, torch.zeros_like(fake_preds))

            d_loss = (real_loss + fake_loss) / 2
            self.log('d_loss', d_loss, prog_bar=True)
            return d_loss

    def configure_optimizers(self):
        g_opt = optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        d_opt = optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        return [g_opt, d_opt], []

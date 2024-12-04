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
            nn.ReLU(),
            nn.Linear(128, 3 * 32 * 32),  # Match CIFAR-10 output dimensions
            nn.Tanh()  # Scale output to [-1, 1]
        )

    def forward(self, z):
        return self.model(z).view(z.size(0), 3, 32, 32)  # Reshape to (batch_size, 3, 32, 32)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3 * 32 * 32, 128),  # Match CIFAR-10 input dimensions
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)  # Flatten image to (batch_size, 3 * 32 * 32)
        return self.model(img_flat)


class GAN(BaseModel):
    def __init__(self, hparams):
        super(GAN, self).__init__(hparams)
        self.lr = hparams["lr"]
        self.latent_dim = hparams["latent_dim"]
        self.generator = Generator(self.latent_dim)
        self.discriminator = Discriminator()
        self.automatic_optimization = False  # Disable automatic optimization

    def forward(self, z):
        return self.generator(z)

    def training_step(self, batch, batch_idx):
        real_imgs, _ = batch
        batch_size = real_imgs.size(0)
        z = torch.randn(batch_size, self.latent_dim, device=self.device)

        # Train Discriminator
        d_opt = self.optimizers()[1]
        d_opt.zero_grad()
        real_preds = self.discriminator(real_imgs)
        real_loss = nn.BCELoss()(real_preds, torch.ones_like(real_preds))
        fake_imgs = self(z).detach()
        fake_preds = self.discriminator(fake_imgs)
        fake_loss = nn.BCELoss()(fake_preds, torch.zeros_like(fake_preds))
        d_loss = (real_loss + fake_loss) / 2
        self.manual_backward(d_loss)
        d_opt.step()

        # Train Generator
        g_opt = self.optimizers()[0]
        g_opt.zero_grad()
        fake_imgs = self(z)
        fake_preds = self.discriminator(fake_imgs)
        g_loss = nn.BCELoss()(fake_preds, torch.ones_like(fake_preds))
        self.manual_backward(g_loss)
        g_opt.step()

        # Logging
        self.log('d_loss', d_loss, prog_bar=True)
        self.log('g_loss', g_loss, prog_bar=True)

    def configure_optimizers(self):
        g_opt = optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        d_opt = optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        return [g_opt, d_opt]

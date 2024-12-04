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
        print(f"Input shape to Generator: {z.shape}")  # Should be [batch_size, latent_dim]
        out = self.model(z)
        print(f"Output shape from Generator before reshape: {out.shape}")  # Should be [batch_size, 3*32*32]
        img = out.view(z.size(0), 3, 32, 32)
        print(f"Output shape from Generator after reshape: {img.shape}")  # Should be [batch_size, 3, 32, 32]
        return img


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
        print(f"Input shape to Discriminator: {img.shape}")  # Should be [batch_size, 3, 32, 32]
        img_flat = img.view(img.size(0), -1)
        print(f"Input shape to Discriminator after flatten: {img_flat.shape}")  # Should be [batch_size, 3*32*32]
        out = self.model(img_flat)
        print(f"Output from Discriminator: {out.shape}")  # Should be [batch_size, 1]
        return out


class GAN(BaseModel):
    def __init__(self, hparams):
        super(GAN, self).__init__(hparams)
        self.latent_dim = hparams["latent_dim"]  # Store latent_dim as an attribute
        print(f"Initialized latent_dim: {self.latent_dim}")
        self.generator = Generator(self.latent_dim)
        self.discriminator = Discriminator()
        self.automatic_optimization = False

    def forward(self, z):
        return self.generator(z)

    def training_step(self, batch, batch_idx):
        print(f"Using latent_dim in training_step: {self.latent_dim}")
        real_imgs, _ = batch
        print(f"Shape of real_imgs: {real_imgs.shape}")
        batch_size = real_imgs.size(0)
        z = torch.randn(batch_size, self.latent_dim, device=self.device)  # Use self.latent_dim
        print(f"Shape of z: {z.shape}")

        # Train Discriminator
        d_opt = self.optimizers()[1]
        d_opt.zero_grad()
        real_preds = self.discriminator(real_imgs)
        real_loss = nn.BCELoss()(real_preds, torch.ones_like(real_preds))
        fake_imgs = self(z).detach()
        print(f"Shape of fake_imgs: {fake_imgs.shape}")
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

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z).view(z.size(0), 1, 28, 28)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        return self.model(img_flat)


class GAN(pl.LightningModule):
    def __init__(self, latent_dim, lr):
        super(GAN, self).__init__()
        self.save_hyperparameters()
        self.generator = Generator(latent_dim)
        self.discriminator = Discriminator()
        self.automatic_optimization = False

    def forward(self, z):
        return self.generator(z)

    def training_step(self, batch, batch_idx):
        real_imgs, _ = batch
        batch_size = real_imgs.size(0)
        z = torch.randn(batch_size, self.hparams.latent_dim, device=self.device)

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
        lr = self.hparams.lr
        g_opt = optim.Adam(self.generator.parameters(), lr=lr)
        d_opt = optim.Adam(self.discriminator.parameters(), lr=lr)
        return [g_opt, d_opt]


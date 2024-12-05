import torch
import torch.nn as nn
from torch import optim
import pytorch_lightning as pl

from src.models.gan import Generator, Discriminator


class AffineCoupling(nn.Module):
    def __init__(self, in_features):
        super(AffineCoupling, self).__init__()
        self.scale_net = nn.Sequential(
            nn.Linear(in_features // 2, 128),
            nn.ReLU(),
            nn.Linear(128, in_features // 2),
            nn.Tanh()  # Scale transformation
        )
        self.translate_net = nn.Sequential(
            nn.Linear(in_features // 2, 128),
            nn.ReLU(),
            nn.Linear(128, in_features // 2)
        )

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)  # Split into two halves
        s = self.scale_net(x1)
        t = self.translate_net(x1)
        y1 = x1
        y2 = x2 * torch.exp(s) + t
        return torch.cat([y1, y2], dim=-1), s.sum(dim=-1)

    def inverse(self, y):
        y1, y2 = y.chunk(2, dim=-1)
        s = self.scale_net(y1)
        t = self.translate_net(y1)
        x1 = y1
        x2 = (y2 - t) * torch.exp(-s)
        return torch.cat([x1, x2], dim=-1)


class NormalizingFlow(nn.Module):
    def __init__(self, in_features, num_layers):
        super(NormalizingFlow, self).__init__()
        self.layers = nn.ModuleList([AffineCoupling(in_features) for _ in range(num_layers)])

    def forward(self, x):
        log_det_jacobians = 0
        for layer in self.layers:
            x, log_det_jacobian = layer(x)
            log_det_jacobians += log_det_jacobian
        return x, log_det_jacobians

    def inverse(self, z):
        for layer in reversed(self.layers):
            z = layer.inverse(z)
        return z


class GANWithFlow(pl.LightningModule):
    def __init__(self, hparams):
        super(GANWithFlow, self).__init__()
        self.save_hyperparameters(hparams)

        # Components
        self.generator = Generator(self.hparams['latent_dim'])
        self.discriminator = Discriminator()
        self.norm_flow = NormalizingFlow(self.hparams['latent_dim'], num_layers=5)

        self.lr = self.hparams['lr']
        self.automatic_optimization = False  # Manual optimization

    def forward(self, z):
        # Apply normalizing flow on latent space
        z, _ = self.norm_flow(z)
        return self.generator(z)

    def training_step(self, batch, batch_idx):
        real_imgs, _ = batch
        batch_size = real_imgs.size(0)
        z = torch.randn(batch_size, self.hparams['latent_dim'], device=self.device)

        # Get optimizers
        g_opt, d_opt = self.optimizers()

        # Train Discriminator
        self.toggle_optimizer(d_opt)
        d_opt.zero_grad()
        real_preds = self.discriminator(real_imgs)
        real_loss = nn.BCELoss()(real_preds, torch.ones_like(real_preds))
        fake_imgs = self(z).detach()
        fake_preds = self.discriminator(fake_imgs)
        fake_loss = nn.BCELoss()(fake_preds, torch.zeros_like(fake_preds))
        d_loss = (real_loss + fake_loss) / 2
        self.manual_backward(d_loss)
        d_opt.step()
        self.untoggle_optimizer(d_opt)

        # Train Generator
        self.toggle_optimizer(g_opt)
        g_opt.zero_grad()
        fake_imgs = self(z)
        fake_preds = self.discriminator(fake_imgs)
        g_loss = nn.BCELoss()(fake_preds, torch.ones_like(fake_preds))
        self.manual_backward(g_loss)
        g_opt.step()
        self.untoggle_optimizer(g_opt)

        # Logging
        self.log('d_loss', d_loss, prog_bar=True)
        self.log('g_loss', g_loss, prog_bar=True)

    def configure_optimizers(self):
        g_opt = optim.Adam(list(self.generator.parameters()) + list(self.norm_flow.parameters()),
                           lr=self.lr, betas=(0.5, 0.999))
        d_opt = optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        return [g_opt, d_opt]


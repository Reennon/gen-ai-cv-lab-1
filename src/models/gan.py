import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import wandb

from src.models.base_model import BaseModel
from torch.autograd import Variable
from pytorch_lightning.loggers import WandbLogger

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
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
        z = Variable(Tensor(np.random.normal(0, 1, (z.shape[0], self.latent_dim))))
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



class GAN(BaseModel):
    def __init__(self, hparams):
        super(GAN, self).__init__(hparams)
        self.generator = Generator(hparams["latent_dim"])
        self.discriminator = Discriminator()
        self.automatic_optimization = False  # Disable automatic optimization

    def forward(self, z):
        return self.generator(z)

    def training_step(self, batch, batch_idx):
        real_imgs, _ = batch
        batch_size = real_imgs.size(0)
        device = real_imgs.device

        # Sample random noise for the generator
        z = torch.randn(batch_size, self.hparams["latent_dim"], device=device)

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

    def validation_step(self, batch, batch_idx):
        real_imgs, _ = batch
        batch_size = real_imgs.size(0)
        device = real_imgs.device

        # Generate fake images
        z = torch.randn(batch_size, self.hparams["latent_dim"], device=device)
        fake_imgs = self(z)

        # Discriminator predictions
        real_preds = self.discriminator(real_imgs)
        fake_preds = self.discriminator(fake_imgs)

        # Calculate validation losses
        real_loss = nn.BCELoss()(real_preds, torch.ones_like(real_preds))
        fake_loss = nn.BCELoss()(fake_preds, torch.zeros_like(fake_preds))
        d_loss = (real_loss + fake_loss) / 2

        g_loss = nn.BCELoss()(fake_preds, torch.ones_like(fake_preds))

        # Logging
        self.log('val_d_loss', d_loss, prog_bar=True)
        self.log('val_g_loss', g_loss, prog_bar=True)

        return {'val_d_loss': d_loss, 'val_g_loss': g_loss}

    def on_validation_epoch_end(self):
        """
        Hook to log generated and real images at the end of the validation epoch.
        Logs real images and generated images as grids to Weights & Biases.
        """
        # Ensure the logger is an instance of WandbLogger
        if isinstance(self.logger, WandbLogger):
            # Generate a batch of random latent vectors
            z = self.sample_latent(batch_size=16, device=self.device)  # Generate 16 random samples
            generated_images = self.generator(z).detach().cpu()

            # Select a few real images from the validation dataset
            real_images = next(iter(self.val_dataloader()))[0][:16].cpu()  # Take the first 16 real images

            # Create grids for visualization
            real_grid = torchvision.utils.make_grid(real_images, nrow=4, normalize=True)
            generated_grid = torchvision.utils.make_grid(generated_images, nrow=4, normalize=True)

            # Log the images to W&B
            self.logger.experiment.log({
                "Real Images": [wandb.Image(real_grid, caption='Real Images')],
                "Generated Images": [wandb.Image(generated_grid, caption='Generated Images')]
            })

    def configure_optimizers(self):
        g_opt = optim.Adam(self.generator.parameters(), lr=self.hparams["lr"], betas=(0.5, 0.999))
        d_opt = optim.Adam(self.discriminator.parameters(), lr=self.hparams["lr"], betas=(0.5, 0.999))
        return [g_opt, d_opt]

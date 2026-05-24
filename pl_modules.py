import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from models import VAE


class plVAE(pl.LightningModule):
    def __init__(self, input_dim=784, latent_dim=32, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = VAE(input_dim, latent_dim)
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)  # Flatten
        x_recon, mu, logvar = self.model(x)

        # Reconstruction loss (Binary Cross Entropy)
        recon_loss = F.binary_cross_entropy(x_recon, x, reduction="sum")

        # KL Divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Total loss
        loss = recon_loss + kl_loss

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)
        x_recon, mu, logvar = self.model(x)

        recon_loss = F.binary_cross_entropy(x_recon, x, reduction="sum")
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kl_loss

        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
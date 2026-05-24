import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from datasets import MNISTDataModule
from pl_modules import plVAE as VAE

input_dim = 784
latent_dim = 32
lr = 1e-3

batch_size = 128
max_epochs = 100
patience = 3

dataset = MNISTDataModule(batch_size=batch_size)
batch = next(iter(dataset.train_dataloader()))

print("Data shape", batch[0].shape)

model = VAE(input_dim=input_dim, latent_dim=latent_dim, lr=lr)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=patience,
    mode="min",
)

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss", mode="min", save_top_k=1, dirpath="./checkpoints/", filename="vae-best-epoch"
)

trainer = pl.Trainer(
    max_epochs=max_epochs,
    accelerator="auto",
    callbacks=[early_stopping, checkpoint_callback],
)

trainer.fit(model, dataset)
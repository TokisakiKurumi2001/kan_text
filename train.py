import argparse

import torch
import wandb
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from TeKan import LiTeKan, TeKANDataLoader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get the language to train adapter")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epoch", type=int, default=10)
    args = parser.parse_args()

    lang = args.lang
    s = args.seed
    lr = args.lr
    seed = seed_everything(s, workers=True)
    epochs = args.epoch  # 50, 100, 150, 250
    torch.set_float32_matmul_precision("high")

    # dataloader
    dataloader_config = {
        "tok_pretrained_ck": "roberta-base",
        "valid_ratio": 0.1,
        "num_train_sample": 10_000,
        "max_length": 256,
    }
    tekan_dataloader = TeKANDataLoader(**dataloader_config)
    [train_dataloader, valid_dataloader] = tekan_dataloader.get_dataloader(batch_size=16, types=["train", "test"])

    wandb_logger = WandbLogger(project=f"tekan", name=f"default", offline=True)
    lr_monitor = LearningRateMonitor(logging_interval="step")

    model_config = {
        "pretrained_ck": "roberta-base",
        "lr": lr, 
    }
    lit_tekan = LiTeKan(**model_config)

    # train model
    trainer = Trainer(
        max_epochs=epochs, devices=[0], accelerator="gpu", logger=wandb_logger, callbacks=[lr_monitor]
    )
    trainer.fit(model=lit_tekan, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)
    lit_tekan.export_model(f"tekan_ckpt")

    wandb.finish()
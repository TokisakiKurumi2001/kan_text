import evaluate
import lightning.pytorch as pl
import torch
import torch.nn as nn

from .configuration import TeKANConfig
from .model import TeKANClassifierModel, TeKAN
from transformers import AutoModel


class LiTeKan(pl.LightningModule):
    def __init__(self, pretrained_ck: str, lr: float):
        super(LiTeKan, self).__init__()
        config = TeKANConfig()
        tekan_model = TeKAN(config)
        pretrained_model = AutoModel.from_pretrained(pretrained_ck)
        self.model = TeKANClassifierModel(pretrained_model, tekan_model)
        self.num_labels = config.num_classes
        self.loss = nn.CrossEntropyLoss()
        self.lr = lr
        self.valid_metric = evaluate.load("metrics/classification_metrics.py")
        self.save_hyperparameters()

    def export_model(self, path):
        self.model.save_pretrained(path)

    def __postprocess(self, predictions, labels):
        predictions = predictions.detach().cpu().clone().numpy()
        labels = labels.detach().cpu().clone().numpy()

        true_labels = labels
        true_predictions = predictions
        return true_predictions, true_labels

    def training_step(self, batch, batch_idx):
        labels = batch.pop("labels")
        logits = self.model(**batch)
        loss = self.loss(logits.view(-1, self.num_labels), labels.view(-1))
        self.log("train/loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        labels = batch.pop("labels")
        logits = self.model(**batch)
        predictions = logits.argmax(dim=-1)

        decoded_preds, decoded_labels = self.__postprocess(predictions, labels)
        self.valid_metric.add_batch(predictions=decoded_preds, references=decoded_labels)

    def on_validation_epoch_end(self):
        results = self.valid_metric.compute()
        self.log("valid/f1", results["f1"], on_epoch=True, on_step=False, sync_dist=True)
        self.log("valid/accuracy", results["accuracy"], on_epoch=True, on_step=False, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, threshold=0.1, mode="max"),
            "monitor": "valid/f1",
            "interval": "epoch",
            "frequency": 1,
            "name": "lr_monitor",
        }
        return [optimizer], [lr_scheduler]
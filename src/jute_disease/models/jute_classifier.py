import torch
import torchmetrics
from lightning import LightningModule
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau


# TODO: Double-check logging for completeness
class JuteClassifier(LightningModule):
    def __init__(
        self,
        feature_extractor: nn.Module,
        num_classes: int = 11,
        lr: float = 1e-3,
        compile_model: bool = True,
    ):
        super().__init__()
        if not hasattr(feature_extractor, "out_features"):
            raise ValueError("Feature extractor must have an 'out_features' attribute")

        self.feature_extractor = feature_extractor
        if compile_model and hasattr(torch, "compile"):
            self.feature_extractor = torch.compile(self.feature_extractor)

        self.classifier = nn.Linear(
            in_features=feature_extractor.out_features, out_features=num_classes
        )
        self.loss = nn.CrossEntropyLoss()

        self.num_classes = num_classes
        self.lr = lr
        self.save_hyperparameters()

        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.test_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(inputs)
        return self.classifier(features)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        y_pred = self(x)
        loss = self.loss(y_pred, y)

        self.log("train_loss", loss, prog_bar=True)
        self.train_acc(y_pred, y)
        self.log(
            "train_acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True
        )

        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        y_pred = self(x)
        loss = self.loss(y_pred, y)

        self.log("val_loss", loss, prog_bar=True)
        self.val_acc(y_pred, y)
        self.val_f1(y_pred, y)
        self.log("val_acc", self.val_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_f1", self.val_f1, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        y_pred = self(x)
        loss = self.loss(y_pred, y)

        self.log("test_loss", loss)
        self.test_acc(y_pred, y)
        self.log("test_acc", self.test_acc)

        return loss

    def predict_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        return self(batch)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer),
                "monitor": "val_loss",
                "frequency": 1,
                "interval": "epoch",
            },
        }

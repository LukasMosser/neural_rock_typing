import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
import pytorch_lightning as pl
from pytorch_lightning.metrics import F1


class NeuralRockModel(pl.LightningModule):
    def __init__(self, ranges, num_classes, N=16):
        super().__init__()
        self.N = N
        self.ranges = ranges
        self.model = models.vgg11(pretrained=True)

        for param in self.model.features.parameters():
            param.requires_grad = False

        self.model.classifier = nn.Sequential(
                                nn.Dropout(p=0.5),
                                nn.Linear(25088, 1024, bias=True),
                                nn.LeakyReLU(inplace=True),
                                nn.Dropout(p=0.5),
                                nn.Linear(1024, 256, bias=True),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(256, num_classes, bias=True))

        self.f1 = F1(num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        preds = self(x)

        loss = F.binary_cross_entropy_with_logits(preds, y)

        self.f1(preds, y)

        self.log('train_loss', loss.item(), on_step=True, on_epoch=True, sync_dist=True)
        self.log('train_f1', self.f1, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.model(x)

        loss = F.binary_cross_entropy_with_logits(preds, y)

        self.f1(preds, y)

        self.log('val_loss', loss.item(), on_step=True, on_epoch=True, sync_dist=True)
        self.log('val_f1', self.f1, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)

        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.classifier.parameters(), lr=3e-4, weight_decay=1e-5)
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
import pytorch_lightning as pl
from pytorch_lightning.metrics import F1


class NeuralRockModel(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()

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

        self.train_f1 = F1(num_classes=num_classes)
        self.val_f1 = F1(num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.squeeze(1)

        logits = self.model(x)
        y_prob = F.log_softmax(logits, dim=1)

        loss = F.cross_entropy(logits, y)

        self.train_f1(y_prob, y)

        return {'loss': loss, 'y': y, 'y_prob': y_prob}

    def training_epoch_end(self, outputs):
        f1 = self.train_f1.compute()

        # Save the metric
        self.log('train_f1_epoch', f1, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.squeeze(1)

        logits = self.model(x)
        y_prob = F.log_softmax(logits, dim=1)

        loss = F.cross_entropy(logits, y)

        self.val_f1(y_prob, y)

        return {'loss': loss, 'y': y, 'y_prob': y_prob}

    def validation_epoch_end(self, outputs):
        accuracy = self.val_f1.compute()

        # Save the metric
        self.log('val_f1_epoch', accuracy, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.classifier.parameters(), lr=3e-4, weight_decay=1e-5)
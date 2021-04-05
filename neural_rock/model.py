import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
import pytorch_lightning as pl
from pytorch_lightning.metrics import F1


class NeuralRockModel(pl.LightningModule):
    def __init__(self, num_classes, learning_rate=3e-4, weight_decay=1e-5, dropout=0.5, average='micro'):
        super().__init__()
        self.save_hyperparameters()

        backbone = models.vgg11(pretrained=True)

        self.feature_extractor = backbone.features
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.classifier = nn.Sequential(nn.Flatten(),
                                nn.Dropout(p=self.dropout),
                                nn.Linear(25088, 1024, bias=True),
                                nn.LeakyReLU(inplace=True),
                                nn.Dropout(p=self.dropout),
                                nn.Linear(1024, 256, bias=True),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(256, num_classes, bias=True))

        self.train_f1 = F1(average=average, num_classes=num_classes)
        self.val_f1 = F1(average=average, num_classes=num_classes)

    def forward(self, x):
        self.feature_extractor.eval()

        with torch.no_grad():
            representations = self.feature_extractor(x)

        x = self.classifier(representations)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.squeeze(1)

        logits = self(x)
        y_prob = F.softmax(logits, dim=1)

        loss = F.cross_entropy(logits, y)

        self.train_f1(y_prob, y)
        self.log('train/loss', loss, on_epoch=True, on_step=False)
        return {'loss': loss, 'y': y, 'y_prob': y_prob}

    def training_epoch_end(self, outputs):
        f1 = self.train_f1.compute()

        # Save the metric
        self.log('train/f1', f1, prog_bar=True, on_epoch=True, on_step=False)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.squeeze(1)

        logits = self(x)
        y_prob = F.softmax(logits, dim=1)

        loss = F.cross_entropy(logits, y)

        self.val_f1(y_prob, y)
        self.log('val/loss', loss, on_epoch=True, on_step=False)
        return {'loss': loss, 'y': y, 'y_prob': y_prob}

    def validation_epoch_end(self, outputs):
        f1 = self.val_f1.compute()

        # Save the metric
        self.log('val/f1', f1, prog_bar=True, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        return torch.optim.Adam(self.classifier.parameters(),
                                lr=self.learning_rate, weight_decay=self.weight_decay)


class NeuralRockModelVGGLinear(NeuralRockModel):
    def __init__(self, *args, **kwargs):
        super(NeuralRockModelVGGLinear, self).__init__(*args, **kwargs)
        self.classifier = nn.Sequential(nn.Flatten(),
                                nn.Dropout(p=self.dropout),
                                nn.Linear(25088, args[0], bias=True))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import pytorch_lightning as pl
from pytorch_lightning.metrics import F1
from neural_rock.networks.lenet import LeNet


class NeuralRockModel(pl.LightningModule):
    def __init__(self, feature_extractor, classifier, num_classes, freeze_feature_extractor=True,
                 learning_rate=3e-4, weight_decay=1e-5, dropout=0.5, average='micro'):
        super().__init__()
        self.save_hyperparameters()

        self.feature_extractor = feature_extractor
        self.classifier = classifier

        if freeze_feature_extractor:
            self.freeze_feature_extractor()

        self.train_f1 = F1(average=average, num_classes=num_classes)
        self.val_f1 = F1(average=average, num_classes=num_classes)

    def freeze_feature_extractor(self):
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, x):
        if self.hparams['freeze_feature_extractor']:
            self.feature_extractor.eval()

            with torch.no_grad():
                representations = self.feature_extractor(x)
        else:
            representations = self.feature_extractor(x)

        x = self.classifier(representations)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch

        logits = self(x)
        y_prob = F.softmax(logits, dim=1)

        loss = F.cross_entropy(logits, y)

        self.train_f1(y_prob, y)
        self.log('train/loss', loss, on_epoch=True, on_step=False)
        return {'loss': loss, 'y': y, 'y_prob': y_prob}

    def training_epoch_end(self, outputs):
        f1 = self.train_f1.compute()

        # Save the metric
        self.log('train/f1', f1, prog_bar=True, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch

        logits = self(x)
        y_prob = F.softmax(logits, dim=1)

        loss = F.cross_entropy(logits, y)

        self.val_f1(y_prob, y)
        self.log('val/loss', loss, on_epoch=True)

        return {'loss': loss, 'y': y, 'y_prob': y_prob}

    def validation_epoch_end(self, outputs):
        f1 = self.val_f1.compute()

        # Save the metric
        self.log('val/f1', f1, prog_bar=True, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        return torch.optim.Adam(self.classifier.parameters() if self.hparams['freeze_feature_extractor'] else self.parameters(),
                                lr=self.hparams['learning_rate'], weight_decay=self.hparams['weight_decay'])


def make_vgg11_model(num_classes, pretrained=True, dropout=0.5):
    backbone = models.vgg11(pretrained=pretrained)
    feature_extractor = backbone.features
    classifier =  nn.Sequential(nn.AdaptiveAvgPool2d(7),
                              nn.Flatten(),
                              nn.Dropout(p=dropout),
                              nn.Linear(25088, 256),
                              nn.LeakyReLU(inplace=True),
                              nn.Dropout(p=dropout),
                              nn.Linear(256, num_classes))
    return feature_extractor, classifier


def make_resnet18_model(num_classes, pretrained=True, **kwargs):
    backbone = models.resnet18(pretrained=pretrained)
    modules = list(backbone.children())[:-1]
    feature_extractor = nn.Sequential(*modules)
    classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, num_classes))
    return feature_extractor, classifier


def make_lenet_model(num_classes):
    backbone = LeNet(N=32, num_classes=num_classes, channels_in=3)

    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight.data)
            torch.nn.init.zeros_(m.bias.data)
        elif isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight.data)
            torch.nn.init.zeros_(m.bias.data)

    backbone.apply(weights_init)

    feature_extractor = backbone.feature_extractor
    classifier = backbone.classifier

    return feature_extractor

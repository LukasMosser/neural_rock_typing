import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import pytorch_lightning as pl
from pytorch_lightning.metrics import F1
from typing import Tuple, Any


class NeuralRockModel(pl.LightningModule):
    """
    Model to train various thin section classifiers.

    Uses F1 score as a base metric.

    Functionality to freeze the feature extractor in the networks
    """
    def __init__(self, feature_extractor: nn.Module,
                 classifier: nn.Module,
                 num_classes: int,
                 freeze_feature_extractor: bool = True,
                 learning_rate: float = 3e-4,
                 weight_decay: float = 1e-5,
                 dropout: float = 0.5,
                 average: str = 'micro'):
        super().__init__()
        self.save_hyperparameters()

        self.feature_extractor = feature_extractor
        self.classifier = classifier

        if freeze_feature_extractor:
            self.freeze_feature_extractor()

        self.train_f1 = F1(average=average, num_classes=num_classes)
        self.val_f1 = F1(average=average, num_classes=num_classes)

    def freeze_feature_extractor(self):
        """
        Set requires grad on the feature_extractors parameters to false if you're not fine-tuning.
        """
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, x: torch.tensor):
        """
        Forward Pass through the network.
        Handles frozen feature extractors by setting eval if flag is set to True.
        """
        if self.hparams['freeze_feature_extractor']:
            self.feature_extractor.eval()

            with torch.no_grad():
                representations = self.feature_extractor(x)
        else:
            representations = self.feature_extractor(x)

        x = self.classifier(representations)
        return x

    def training_step(self,
                      batch: Tuple[torch.Tensor],
                      batch_idx: int):
        """
        Training Step for Classification Model and Logging to Tensorbaord
        """
        x, y = batch

        logits = self(x)
        y_prob = F.softmax(logits, dim=1)

        loss = F.cross_entropy(logits, y)

        self.train_f1(y_prob, y)
        self.log('train/loss', loss, on_epoch=True, on_step=False)
        return {'loss': loss, 'y': y, 'y_prob': y_prob}

    def training_epoch_end(self,
                           outputs: Tuple[Any]):
        """
        Colate results and log at the end of each training epoch.
        """
        f1 = self.train_f1.compute()

        # Save the metric
        self.log('train/f1', f1, prog_bar=True, on_epoch=True)

    def validation_step(self,
                        batch: Tuple[torch.tensor],
                        batch_idx: int):
        """
        Validation Step for Classification Model and Logging to Tensorbaord
        """

        x, y = batch

        logits = self(x)
        y_prob = F.softmax(logits, dim=1)

        loss = F.cross_entropy(logits, y)

        self.val_f1(y_prob, y)
        self.log('val/loss', loss, on_epoch=True)

        return {'loss': loss, 'y': y, 'y_prob': y_prob}

    def validation_epoch_end(self,
                             outputs: Tuple[Any]):
        """
        Colate results and log at the end of each training epoch.
        """
        f1 = self.val_f1.compute()

        # Save the metric
        self.log('val/f1', f1, prog_bar=True, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        """
        Setup the optimizers for the training process.
        Only optimizes classifier if freeze_feature_extractor is set to True.
        else entire model is being fine-tuned
        """
        return torch.optim.Adam(self.classifier.parameters() if self.hparams['freeze_feature_extractor'] else self.parameters(),
                                lr=self.hparams['learning_rate'], weight_decay=self.hparams['weight_decay'])


def make_vgg11_model(num_classes: int,
                     pretrained: bool = True,
                     dropout: float = 0.5) -> Tuple[nn.Module]:
    """
    Create a feature extractor and classifier from pretrained vgg11
    """
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


def make_resnet18_model(num_classes: int,
                        pretrained: bool = True,
                        **kwargs) -> Tuple[nn.Module]:
    """
    Create a feature extractor and classifier from pretrained resnet18
    """
    backbone = models.resnet18(pretrained=pretrained)
    modules = list(backbone.children())[:-1]
    feature_extractor = nn.Sequential(*modules)
    classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, num_classes))
    return feature_extractor, classifier

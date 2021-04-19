import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNetFeatureExtractor(nn.Module):
    def __init__(self, N=32, channels_in=3, activation=F.relu):
        super(LeNetFeatureExtractor, self).__init__()
        self.act = activation
        self.conv1 = nn.Conv2d(channels_in, N, (5, 5), padding=2)
        self.conv2 = nn.Conv2d(N, 2*N, (5, 5))
        self.conv3 = nn.Conv2d(2*N, 4*N, (5, 5))
        self.bn1 = nn.BatchNorm2d(N)
        self.bn2 = nn.BatchNorm2d(2*N)
        self.bn3 = nn.BatchNorm2d(4*N)

    def forward(self, x):
        x1 = self.act(self.bn1(self.conv1(x)))
        x2 = self.act(self.bn2(self.conv2(x1)))
        x3 = self.act(self.bn3(self.conv3(x2)))
        return x3


class LeNet(nn.Module):
    def __init__(self, N=32, channels_in=3, num_classes=1, activation=F.relu):
        super(LeNet, self).__init__()
        self.feature_extractor = LeNetFeatureExtractor(N=N, channels_in=channels_in, activation=activation)
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(5), nn.Flatten(),
                                        nn.Linear(4*N*5*5, 256), nn.ReLU(), nn.Linear(256, num_classes))

    def forward(self, x):
        features = self.feature_extractor(x)
        out = self.classifier(features)
        return out
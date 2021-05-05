import numpy as np
import torch
from neural_rock.dataset import ThinSectionDataset
from neural_rock.model import NeuralRockModel, make_vgg11_model
import albumentations as A
import torch.nn as nn
from torchvision import transforms
from pathlib import Path
from torch.autograd import Variable

MEAN_TRAIN = np.array([0.485, 0.456, 0.406])
STD_TRAIN = np.array([0.229, 0.224, 0.225])


class Model(nn.Module):
    def __init__(self, feature_extractor, fc):
        super(Model, self).__init__()
        self.feature_extractor = feature_extractor
        self.fc = fc

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.fc(x)
        return x


def load_data(labelset):
    transform = A.Compose([A.Normalize()])

    train_dataset = ThinSectionDataset(Path("."), labelset,
                                       transform=transform, train=True, seed=42, preload_images=False)
    val_dataset = ThinSectionDataset(Path("."), labelset,
                                     transform=transform, train=False, seed=42, preload_images=False)

    #assert not any([train_id in val_dataset.image_ids for train_id in train_dataset.image_ids])

    #modified_label_map = train_dataset.modified_label_map
    #image_names = train_dataset.image_ids
    #class_names = train_dataset.class_names

    paths = {}
    for train, dataset in zip([True, False], [train_dataset.dataset, val_dataset.dataset]):
        for key, dset in dataset.items():
            paths[key] = {'path': dset['path'],
                          'label': dset[labelset],
                          'mask_path': dset['mask_path'],
                          'train': "Train" if train else "Test"}

    return paths


def load_model(checkpoint, num_classes=5, model_init_func=make_vgg11_model):
    feature_extractor, classifier = model_init_func(num_classes=num_classes)
    model = NeuralRockModel(feature_extractor, classifier, num_classes=num_classes).load_from_checkpoint(checkpoint)
    model.eval()
    model.freeze()
    return model


def compute_images(X, grad_cam, max_classes, resize, device="cpu"):
    maps = []
    for i in range(max_classes):
        gb = grad_cam(X.to(device), i)
        gb = (gb - np.min(gb)) / (np.max(gb) - np.min(gb))
        gb = resize(torch.from_numpy(gb).unsqueeze(0))[0]
        maps.append(gb.cpu().numpy())
    maps = np.stack(maps, axis=0)
    return maps


def make_cam_map(X, grad_cam, i, device='cpu', ratio=224.0/512.0):
    transform = A.Compose([
        A.Resize(int(ratio * X.shape[0]), int(ratio * X.shape[1])),
        A.Normalize()])

    X = transform(image=X)['image'].transpose(2, 0, 1)
    X = Variable(torch.from_numpy(X).unsqueeze(0), requires_grad=True)
    X = X.to(device)
    cam_map = compute_cam(X, grad_cam, i, device=device)
    return cam_map


def compute_cam(X, grad_cam, i, device="cpu"):
    gb = grad_cam(X, i)
    gb = (gb - np.min(gb)) / (np.max(gb) - np.min(gb))
    return gb

import os
import pandas as pd
import numpy as np

import copy
import time
from PIL import Image, ImageFilter

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.cm as mpl_color_map

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import TensorDataset
from torch.nn import ReLU
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

import imageio
import h5py
from tqdm import tqdm
import neural_rock.preprocess as pre
from neural_rock.dataset import split_dataset, ThinSectionDataset
from neural_rock.training import train, validate
from neural_rock.utils import set_seed, save_checkpoint, create_run_directory, get_lr

set_seed(42, cudnn=True, benchmark=True)

path = create_run_directory("./runs")

train_step_writer = SummaryWriter(log_dir=os.path.join(path, "tensorboard", "train"))
val_step_writer = SummaryWriter(log_dir=os.path.join(path, "tensorboard", "val"))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

df = pre.load_excel()

imgs, features, df_ = pre.load_images_w_rows_in_table(df)
imgs = np.array(imgs)

pore_type, modified_label_map, class_names = pre.make_feature_map_dunham(features)
pore_type = np.array(pore_type)

splitter = StratifiedShuffleSplit(test_size=0.50, random_state=42)

for train_idx, val_idx in splitter.split(imgs, pore_type):
    print(train_idx, val_idx)
    break

imgs_train, pore_type_train = imgs[train_idx], pore_type[train_idx]
imgs_val, pore_type_val = imgs[val_idx], pore_type[val_idx]

print(np.unique(pore_type_train, return_counts=True))
print(np.unique(pore_type_val, return_counts=True))

del imgs

X_train_np, y_train_np = pre.create_images_and_labels(imgs_train, pore_type_train)
X_val_np, y_val_np = pre.create_images_and_labels(imgs_val, pore_type_val)

mean_train = np.mean(X_train_np, axis=(0, 2, 3))
std_train = np.std(X_train_np, axis=(0, 2, 3))

print(mean_train, std_train)

del imgs_train
del imgs_val

data_transforms = {
    'train': transforms.Compose([
        transforms.Normalize(mean_train, std_train),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomCrop(750),
        transforms.RandomRotation([0, 360]),
        transforms.CenterCrop(500),
        transforms.Resize(224),
    ]),
    'val': transforms.Compose([
        transforms.Normalize(mean_train, std_train),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomCrop(750),
        transforms.RandomRotation([0, 360]),
        transforms.CenterCrop(500),
        transforms.Resize(224),
    ]),
}

train_dataset = ThinSectionDataset(torch.from_numpy(X_train_np), torch.from_numpy(y_train_np), class_names, transform=data_transforms['train'])
val_dataset = ThinSectionDataset(torch.from_numpy(X_val_np), torch.from_numpy(y_val_np), class_names, transform=data_transforms['val'])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

model = models.vgg11(pretrained=True)

for param in model.features.parameters():
    param.requires_grad = False

print(model)
model.classifier = nn.Sequential(
            nn.Dropout(p=0.9),
            nn.Linear(25088, 128),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, 64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, len(class_names)))

model = model.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer = optim.SGD(model.classifier.parameters(), lr=0.001, momentum=0.9)

max_steps = 5000

# Decay LR by a factor of 0.1 every 7 epochs
#exp_lr_scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=1e-4, max_lr=0.01, step_size_up=1000)

best_f1 = 0.0
validate_every = 50
sgd_steps = 0
initial = True

with tqdm(total=max_steps+1) as pbar:
    while sgd_steps < max_steps:
        sgd_steps, epoch_train_loss, epoch_train_f1 = train(sgd_steps, model, criterion, optimizer, train_loader, train_step_writer, device=device)

        train_step_writer.add_scalar("epoch_f1", global_step=sgd_steps, scalar_value=epoch_train_f1)
        train_step_writer.add_scalar("epoch_loss", global_step=sgd_steps, scalar_value=epoch_train_loss)
        train_step_writer.add_scalar("learning_rate", global_step=sgd_steps, scalar_value=get_lr(optimizer))
        pbar.update(1)

        if sgd_steps % validate_every == 0 or initial:
            epoch_val_loss, epoch_val_f1, predictions = validate(model, criterion, val_loader, device="cuda", return_predictions=True)
            val_step_writer.add_scalar("epoch_f1", global_step=sgd_steps, scalar_value=epoch_val_f1)
            val_step_writer.add_scalar("f1_score", global_step=sgd_steps, scalar_value=epoch_val_f1)
            val_step_writer.add_scalar("epoch_loss", global_step=sgd_steps, scalar_value=epoch_val_loss)
            val_step_writer.add_scalar("loss", global_step=sgd_steps, scalar_value=epoch_val_loss)
            initial = False

            save_checkpoint(sgd_steps, model, optimizer, os.path.join(path, "checkpoints"), is_best=epoch_val_f1 > best_f1)
            if epoch_val_f1 > best_f1:
                best_f1 = epoch_val_f1

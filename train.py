import sys
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import models

import neural_rock.preprocess as pre
from neural_rock.dataset import ThinSectionDataset
from neural_rock.training import train, validate
from neural_rock.utils import set_seed, save_checkpoint, create_run_directory, get_lr
from neural_rock.model import NeuralRockModel

def visualize_batch(loader, std, mean):
    fig, axarr = plt.subplots(4, 4, figsize=(12, 12))
    for dat, _ in loader:
        break
    for ax, im in zip(axarr.flatten(), dat.numpy()):
        im = im.transpose(1, 2, 0)*std+mean
        ax.imshow(im)
    plt.show()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="MLE", choices=["Dunham", "DominantPore", "Lucia"])
    parser.add_argument("--base_dir", type=str, default="./runs")
    parser.add_argument("--wd", type=float, default=1e-4, help="Which batch_size to use for training")
    parser.add_argument("--lr_init", type=float, default=1e-3, help="Which batch_size to use for training")
    parser.add_argument("--momentum", type=float, default=0.9, help="Which batch_size to use for training")
    parser.add_argument("--dropout", type=float, default=0.5, help="Which batch_size to use for training")
    parser.add_argument("--val_split_size", type=float, default=0.5, help="Which batch_size to use for training")
    parser.add_argument("--optimizer", type=str, default="SGD", choices=["SGD"])
    parser.add_argument("--num_workers", type=int, default=4, help="How many workers to use")
    parser.add_argument("--batch_size", type=int, default=16, help="Which batch_size to use for training")
    parser.add_argument("--smoketest", action="store_true", help="Which batch_size to use for training")
    parser.add_argument("--epochs", type=int, default=200, help="Which batch_size to use for training")
    parser.add_argument("--seed", type=int, default=42, help="Which batch_size to use for training")
    parser.add_argument("--store_freq", type=int, default=20, help="How often to store checkpoints")
    args = parser.parse_args(argv)

    args.swa = False
    if args.method == "SWAG":
        args.swa = True
    return args


def main(args):
    set_seed(42, cudnn=True, benchmark=True)

    path = create_run_directory("./runs", prefix=args.method)

    train_step_writer = SummaryWriter(log_dir=os.path.join(path, "tensorboard", "train"))
    val_step_writer = SummaryWriter(log_dir=os.path.join(path, "tensorboard", "val"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    data_transforms = {
        'train': A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Rotate(360, always_apply=True),
                A.RandomCrop(width=512, height=512),
                A.ElasticTransform(sigma=25, alpha_affine=25),
                A.GaussNoise(),
                A.HueSaturationValue(sat_shift_limit=0, val_shift_limit=50, hue_shift_limit=255, always_apply=True),
                A.Resize(width=224, height=224),
                A.Normalize(mean=mean, std=std)
    ]),
        'val': A.Compose([
        A.RandomCrop(width=512, height=512),
        A.Resize(width=224, height=224),
        A.Normalize(mean=mean, std=std),
        ])
    }

    train_dataset = ThinSectionDataset("./data/Images_PhD_Miami/Leg194", args.method,
                                       transform=data_transforms['train'], train=True, seed=args.seed)
    val_dataset = ThinSectionDataset("./data/Images_PhD_Miami/Leg194", args.method,
                                     transform=data_transforms['val'], train=False, seed=args.seed)

    idx_check = [train_id in val_dataset.image_ids for train_id in train_dataset.image_ids]
    assert not any(idx_check)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=10)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=10)
    visualize_batch(train_loader, mean, std)
    visualize_batch(val_loader, mean, std)
    len(train_dataset.class_names)

    criterion = nn.CrossEntropyLoss()#weight=train_dataset.weights.float().to(device))

    # Observe that all parameters are being optimized
    optimizer = optim.Adam(model.classifier.parameters(), lr=3e-4)

    max_steps = 5000

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
                val_step_writer.add_scalar("epoch_loss", global_step=sgd_steps, scalar_value=epoch_val_loss)
                val_step_writer.add_scalar("loss", global_step=sgd_steps, scalar_value=epoch_val_loss)
                initial = False

                save_checkpoint(sgd_steps, model, optimizer, os.path.join(path, "checkpoints"), is_best=epoch_val_f1 > best_f1)
                if epoch_val_f1 > best_f1:
                    best_f1 = epoch_val_f1

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)

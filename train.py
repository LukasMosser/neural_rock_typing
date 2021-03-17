import sys
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm
import neural_rock.preprocess as pre
from neural_rock.dataset import ThinSectionDataset
from neural_rock.training import train, validate
from neural_rock.utils import set_seed, save_checkpoint, create_run_directory, get_lr


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="MLE", choices=["Dunham", "DominantPore", "Lucia"])
    parser.add_argument("--base_dir", type=str, default="./runs")
    parser.add_argument("--wd", type=float, default=1e-4, help="Which batch_size to use for training")
    parser.add_argument("--lr_init", type=float, default=1e-3, help="Which batch_size to use for training")
    parser.add_argument("--momentum", type=float, default=0.9, help="Which batch_size to use for training")
    parser.add_argument("--dropout", type=float, default=0.5, help="Which batch_size to use for training")
    parser.add_argument("--optimizer", type=str, default="SGD", choices=["SGD"])
    parser.add_argument("--num_workers", type=int, default=4, help="How many workers to use")
    parser.add_argument("--batch_size", type=int, default=16, help="Which batch_size to use for training")
    parser.add_argument("--smoketest", action="store_true", help="Which batch_size to use for training")
    parser.add_argument("--epochs", type=int, default=200, help="Which batch_size to use for training")
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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    df = pre.load_excel()

    imgs, features, df_ = pre.load_images_w_rows_in_table(df)
    imgs = np.array(imgs)*255
    if args.method == "Lucia":
        pore_type, modified_label_map, class_names = pre.make_feature_map_lucia(features)
    elif args.method == "DominantPore":
        pore_type, modified_label_map, class_names = pre.make_feature_map_microp(features)
    else:
        pore_type, modified_label_map, class_names = pre.make_feature_map_dunham(features)
    pore_type = np.array(pore_type)

    splitter = StratifiedShuffleSplit(test_size=0.50, random_state=42)

    for train_idx, val_idx in splitter.split(imgs, pore_type):
        print(train_idx, val_idx)
        break

    imgs_train, pore_type_train = imgs[train_idx], pore_type[train_idx]
    imgs_val, pore_type_val = imgs[val_idx], pore_type[val_idx]

    print("train", np.unique(pore_type_train, return_counts=True))
    print("val", np.unique(pore_type_val, return_counts=True))

    _, class_frequency = np.unique(pore_type_train, return_counts=True)
    class_frequency = class_frequency/np.sum(class_frequency)

    weights = torch.from_numpy(1./class_frequency)
    print(weights)
    print(np.unique(pore_type_val, return_counts=True))

    del imgs

    X_train_np, y_train_np = pre.create_images_and_labels(imgs_train, pore_type_train)
    X_val_np, y_val_np = pre.create_images_and_labels(imgs_val, pore_type_val)

    X_train_np = X_train_np.astype(np.uint8)
    X_val_np = X_val_np.astype(np.uint8)

    mean_train = np.mean(X_train_np, axis=(0, 2, 3))
    std_train = np.std(X_train_np, axis=(0, 2, 3))

    print(mean_train, std_train)

    del imgs_train
    del imgs_val

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

    train_dataset = ThinSectionDataset(torch.from_numpy(X_train_np), torch.from_numpy(y_train_np), class_names, transform=data_transforms['train'])
    val_dataset = ThinSectionDataset(torch.from_numpy(X_val_np), torch.from_numpy(y_val_np), class_names, transform=data_transforms['val'])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    model = models.vgg11(pretrained=True)

    fig, axarr = plt.subplots(4, 4, figsize=(12, 12))

    for dat, _ in train_loader:
        break

    for ax, im in zip(axarr.flatten(), dat.numpy()):
        im = im.transpose(1, 2, 0)*std+mean
        ax.imshow(im)
    plt.show()

    for param in model.features.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(25088, 1024, bias=True),
                nn.LeakyReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(1024, 256, bias=True),
                nn.LeakyReLU(inplace=True),
                nn.Linear(256, len(class_names), bias=True))

    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=weights.float().to(device))

    # Observe that all parameters are being optimized
    optimizer = optim.SGD(model.classifier.parameters(), lr=0.001, momentum=0.9)

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
                val_step_writer.add_scalar("f1_score", global_step=sgd_steps, scalar_value=epoch_val_f1)
                val_step_writer.add_scalar("epoch_loss", global_step=sgd_steps, scalar_value=epoch_val_loss)
                val_step_writer.add_scalar("loss", global_step=sgd_steps, scalar_value=epoch_val_loss)
                initial = False

                save_checkpoint(sgd_steps, model, optimizer, os.path.join(path, "checkpoints"), is_best=epoch_val_f1 > best_f1)
                if epoch_val_f1 > best_f1:
                    best_f1 = epoch_val_f1

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)

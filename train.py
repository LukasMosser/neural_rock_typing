import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from neural_rock.dataset import ThinSectionDataset
from neural_rock.utils import set_seed, create_run_directory
from neural_rock.model import NeuralRockModel


def visualize_batch(loader):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

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

    data_transforms = {
        'train': A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Rotate(360, always_apply=True),
                A.RandomCrop(width=512, height=512),
                A.ElasticTransform(sigma=25, alpha_affine=25),
                A.GaussNoise(),
                A.HueSaturationValue(sat_shift_limit=0, val_shift_limit=50, hue_shift_limit=255, always_apply=True),
                A.Resize(width=224, height=224),
                A.Normalize()
    ]),
        'val': A.Compose([
        A.RandomCrop(width=512, height=512),
        A.Resize(width=224, height=224),
        A.Normalize(),
        ])
    }

    train_dataset = ThinSectionDataset("./data/Images_PhD_Miami/Leg194", args.method,
                                       transform=data_transforms['train'], train=True, seed=args.seed)
    val_dataset = ThinSectionDataset("./data/Images_PhD_Miami/Leg194", args.method,
                                     transform=data_transforms['val'], train=False, seed=args.seed)

    idx_check = [train_id in val_dataset.image_ids for train_id in train_dataset.image_ids]
    assert not any(idx_check)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=10)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=10)

    visualize_batch(train_loader)
    visualize_batch(val_loader)

    trainer = pl.Trainer(gpus=-1, max_epochs=1000, benchmark=True, check_val_every_n_epoch=10)

    model = NeuralRockModel(num_classes=len(train_dataset.class_names))

    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)

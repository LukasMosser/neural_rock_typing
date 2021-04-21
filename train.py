import sys
import argparse
import albumentations as A
from torch.utils.data import DataLoader, ConcatDataset
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from neural_rock.dataset import ThinSectionDataset
from neural_rock.utils import set_seed
from neural_rock.model import NeuralRockModel, NeuralRockModelResnetFC, NeuralRockModeLeNetFC
from neural_rock.plot import visualize_batch


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--labelset", type=str, default="Dunham", choices=["Dunham", "DominantPore", "Lucia"])
    parser.add_argument("--model", type=str, default=['VGG_FC'], choices=['VGG_FC', 'VGG_Linear', 'Resnet', 'LeNet'])
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Which batch_size to use for training")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Which batch_size to use for training")
    parser.add_argument("--dropout", type=float, default=0.5, help="Which batch_size to use for training")
    parser.add_argument("--val_split_size", type=float, default=0.5, help="Which batch_size to use for training")
    parser.add_argument("--num_workers", type=int, default=4, help="How many workers to use")
    parser.add_argument("--num_val", type=int, default=50, help="How many workers to use")
    parser.add_argument("--batch_size", type=int, default=16, help="Which batch_size to use for training")
    parser.add_argument("--smoketest", action="store_true", help="Which batch_size to use for training")
    parser.add_argument("--plot", action="store_true", help="Which batch_size to use for training")
    parser.add_argument("--epochs", type=int, default=100, help="Which batch_size to use for training")
    parser.add_argument("--seed", type=int, default=42, help="Which batch_size to use for training")
    args = parser.parse_args(argv)

    return args


def main(args):

    set_seed(42, cudnn=True, benchmark=True)

    data_transforms = {
        'train': A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Rotate(360, always_apply=True),
                A.RandomCrop(width=512, height=512),
                A.HueSaturationValue(sat_shift_limit=0, val_shift_limit=0, hue_shift_limit=255, always_apply=True),
                A.Resize(width=224, height=224),
                A.Normalize()
    ]),
        'val': A.Compose([
        A.RandomCrop(width=512, height=512),
        A.Resize(width=224, height=224),
        A.Normalize(),
        ])
    }

    train_dataset_base = ThinSectionDataset("./data/Images_PhD_Miami/Leg194", args.labelset,
                                       transform=data_transforms['train'], train=True, seed=args.seed)

    print(train_dataset_base.image_ids)

    val_dataset = ThinSectionDataset("./data/Images_PhD_Miami/Leg194", args.labelset,
                                     transform=data_transforms['val'], train=False, seed=args.seed)
    print(val_dataset.image_ids)
    train_dataset = ConcatDataset([train_dataset_base]*10)
    val_dataset = ConcatDataset([val_dataset]*args.num_val)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, prefetch_factor=10)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, prefetch_factor=10)

    if args.plot:
        visualize_batch(train_loader)
        visualize_batch(val_loader)

    wandb_logger = WandbLogger(name='lukas-mosser', project='neural-rock', entity='ccg')
    wandb_logger.experiment.config.update(args)

    checkpointer = ModelCheckpoint(monitor="val/f1", verbose=True, mode="max")
    trainer = pl.Trainer(gpus=-1, max_epochs=args.epochs, benchmark=True,
                         logger=[wandb_logger],
                         callbacks=[checkpointer],
                         check_val_every_n_epoch=1)
    if args.model == 'Resnet':
        model = NeuralRockModelResnetFC(len(train_dataset_base.class_names))
    elif args.model == 'LeNet':
        model = NeuralRockModeLeNetFC(len(train_dataset_base.class_names))
    else:
        model = NeuralRockModel(len(train_dataset_base.class_names))

    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)

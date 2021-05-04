import sys
import argparse
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from neural_rock.dataset import GPUThinSectionDataset, ThinSectionDataset
from neural_rock.utils import MEAN_TRAIN, STD_TRAIN
from neural_rock.model import NeuralRockModel, make_vgg11_model, make_lenet_model, make_resnet18_model
from neural_rock.plot import visualize_batch
from torchvision import transforms
from pathlib import Path

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--labelset", type=str, default="Dunham", choices=["Dunham", "DominantPore", "Lucia"])
    parser.add_argument("--model", type=str, default=['VGG_FC'], choices=['VGG_FC', 'VGG_Linear', 'Resnet', 'LeNet'])
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Which batch_size to use for training")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Which batch_size to use for training")
    parser.add_argument("--dropout", type=float, default=0.5, help="Which batch_size to use for training")
    parser.add_argument("--val_split_size", type=float, default=0.5, help="Which batch_size to use for training")
    parser.add_argument("--num_workers", type=int, default=4, help="How many workers to use")
    parser.add_argument("--batch_size", type=int, default=16, help="Which batch_size to use for training")
    parser.add_argument("--seed", type=int, default=42, help="Which batch_size to use for training")
    parser.add_argument("--test_batch_size", type=int, default=8, help="Which batch_size to use for training")
    parser.add_argument("--accumulate_batches", type=int, default=1, help="Which batch_size to use for training")
    parser.add_argument("--steps", type=int, default=10000, help="Which batch_size to use for training")
    parser.add_argument("--log_every_n_steps", type=int, default=1, help="Which batch_size to use for training")
    parser.add_argument("--benchmark", action='store_true', help="Use CUDA Benchmark mode")
    parser.add_argument("--pin_memory", action='store_true', help="Use CUDA Benchmark mode")
    parser.add_argument("--plot", action='store_true', help="Use CUDA Benchmark mode")
    parser.add_argument("--precision", type=int, default=32, choices=[16, 32], help="Use CUDA Benchmark mode")
    parser.add_argument("--distributed_backend")
    args = parser.parse_args(argv)

    return args


def main(args):
    pl.seed_everything(args.seed)

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=360),
            transforms.RandomCrop((512, 512)),
            transforms.ColorJitter(hue=0.5),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=MEAN_TRAIN, std=STD_TRAIN)
        ]),
        'val':
            transforms.Compose([
                transforms.CenterCrop((512, 512)),
                transforms.Resize((224, 224)),
                transforms.Normalize(mean=MEAN_TRAIN, std=STD_TRAIN)
            ])
    }

    base_path = Path(".")

    train_dataset_base = ThinSectionDataset(base_path, args.labelset,
                                            preload_images=True,
                                            transform=data_transforms['train'], train=True, seed=args.seed)

    val_dataset = ThinSectionDataset(base_path, args.labelset, preload_images=True,
                                       transform=data_transforms['val'], train=False, seed=args.seed)

    train_loader = DataLoader(train_dataset_base, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=args.pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)

    if args.plot:
        visualize_batch(train_loader)
        visualize_batch(val_loader)

    wandb_logger = WandbLogger(name='lukas-mosser', project='neural-rock', entity='ccg')
    wandb_logger.experiment.config.update(args)

    checkpointer = ModelCheckpoint(monitor="val/f1", verbose=True, mode="max")
    trainer = pl.Trainer(gpus=-1,
                         max_epochs=None,
                         logger=[wandb_logger],
                         callbacks=[checkpointer],
                         log_every_n_steps=args.log_every_n_steps,
                         distributed_backend=args.distributed_backend,
                         max_steps=args.steps,
                         benchmark=args.benchmark)

    feature_extractor, classifier = make_vgg11_model(num_classes=train_dataset_base.num_classes, dropout=args.dropout)
    model = NeuralRockModel(feature_extractor, classifier, train_dataset_base.num_classes)

    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)

import sys
from pathlib import Path
import argparse
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from neural_rock.dataset import ThinSectionDataset
from neural_rock.utils import MEAN_TRAIN, STD_TRAIN
from neural_rock.model import NeuralRockModel, make_vgg11_model, make_resnet18_model
from neural_rock.plot import visualize_batch


def parse_arguments(argv):
    """
    Parse Command Line Arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--labelset", type=str, default="Dunham", choices=["Dunham", "DominantPore", "Lucia"], help="Which labelset do you want to train on?")
    parser.add_argument("--model", type=str, default=['vgg'], choices=['vgg','resnet'], help="Which model type do you want to use?")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight Decay Parameter")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning Rate Parameter")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout Value")
    parser.add_argument("--val_split_size", type=float, default=0.5, help="Train test split size used for training.")
    parser.add_argument("--num_workers", type=int, default=4, help="How many workers to use")
    parser.add_argument("--batch_size", type=int, default=16, help="Which batch_size to use for training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed to use.")
    parser.add_argument("--test_batch_size", type=int, default=8, help="Batch size used when performing validation and testing.")
    parser.add_argument("--accumulate_batches", type=int, default=1, help="How many batches to accumulate before performing gradient update.")
    parser.add_argument("--steps", type=int, default=10000, help="How many gradient steps to perform, because epochs are meaningless.")
    parser.add_argument("--log_every_n_steps", type=int, default=1, help="How many steps until something gets logged.")
    parser.add_argument("--benchmark", action='store_true', help="Use CUDA Benchmark mode")
    parser.add_argument("--freeze_feature_extractor", action='store_true', help="Do you want to freeze the feature extractor?")
    parser.add_argument("--pin_memory", action='store_true', help="Pin Memory")
    parser.add_argument("--plot", action='store_true', help="Plot some batches?")
    parser.add_argument("--precision", type=int, default=32, choices=[16, 32], help="Which precision mode to use")
    parser.add_argument("--distributed_backend", help="Got too many GPUs?")
    args = parser.parse_args(argv)

    return args


def main(args):
    # We first set the Random Seed for Everything.
    # Recommended to use seed=42 as that will give best results according to Douglas Adams
    pl.seed_everything(args.seed)

    # Data Augmentation Pipeline
    # Uses HSV color jitter to focus network on textural rather than color features.
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(), # Do Random Flip
            transforms.RandomRotation(degrees=360), # Rotate the image as orientation doesn't matter
            transforms.RandomCrop((512, 512)), # Crop to a smaller size randomly
            transforms.ColorJitter(hue=0.5), # Color Jitter the hue
            transforms.Resize((224, 224)), # Resize to Resnet or VGG input Size
            transforms.Normalize(mean=MEAN_TRAIN, std=STD_TRAIN) # Normalize to Imagenet Pixel Value Distribution
        ]),
        'val':
            transforms.Compose([
                transforms.CenterCrop((512, 512)), # Center Crop
                transforms.Resize((224, 224)), # Resize
                transforms.Normalize(mean=MEAN_TRAIN, std=STD_TRAIN) # Normalize to pretrained imagenet weights
            ])
    }

    # Set base working path
    base_path = Path("..")

    # Initialize the training and Validation Datasets
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

    # Use the Weights And Biases Logger
    wandb_logger = WandbLogger(name='lukas-mosser', project='neural-rock', entity='ccg')
    wandb_logger.experiment.config.update(args)

    # Checkpoint Models based on Validation F1 score
    checkpointer = ModelCheckpoint(monitor="val/f1", verbose=True, mode="max")

    # Initialize the Pytorch Lightning Trainer
    trainer = pl.Trainer(gpus=-1,
                         max_epochs=None,
                         logger=[wandb_logger],
                         callbacks=[checkpointer],
                         log_every_n_steps=args.log_every_n_steps,
                         distributed_backend=args.distributed_backend,
                         max_steps=args.steps,
                         benchmark=args.benchmark)

    # Make the Model
    if args.model == 'vgg':
        feature_extractor, classifier = make_vgg11_model(num_classes=train_dataset_base.num_classes,
                                                         dropout=args.dropout)

    elif args.model == 'resnet':
        feature_extractor, classifier = make_resnet18_model(num_classes=train_dataset_base.num_classes,
                                                            dropout=args.dropout)

    # Intialize the Neural Rock Model with the feature extractor and classifier
    model = NeuralRockModel(feature_extractor,
                            classifier,
                            train_dataset_base.num_classes,
                            freeze_feature_extractor=args.freeze_feature_extractor)

    # Run actual training
    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)

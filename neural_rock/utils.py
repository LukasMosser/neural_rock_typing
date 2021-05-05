import json
import random
import torch
import numpy as np
import os
from datetime import datetime
import shutil
import errno


def create_run_directory(output_path_directory, prefix=""):
    mydir = os.path.join(output_path_directory, "_".join([prefix, datetime.now().strftime('%Y-%m-%d_%H-%M-%S')]))

    for dir in [mydir, os.path.join(mydir, "checkpoints"), os.path.join(mydir, "tensorboard")]:
        try:
            os.makedirs(dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise  # This was not a "directory exist" error..
    return mydir


def set_seed(seed, cudnn=True, benchmark=True):
    """
    Use this to set ALL the random seeds to a fixed value and take out any randomness from cuda kernels
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = benchmark
    torch.backends.cudnn.enabled = cudnn

    return True


def save_checkpoint(epoch, model, optimizer, path, is_best=False):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(path, "epoch_{0:}".format(epoch) + ".pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, filepath)

    if is_best:
        shutil.copyfile(filepath, os.path.join(path, 'best.pth'))


def load_checkpoint(path, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(path):
        raise("File doesn't exist {}".format(path))
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


MEAN_TRAIN = [0.485, 0.456, 0.406]
STD_TRAIN = [0.229, 0.224, 0.225]


def get_train_test_split(path="./data/train_test_split.json"):
    with open(path) as f:
        train_test_split = json.load(f)
    return train_test_split
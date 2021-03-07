import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedShuffleSplit


class ThinSectionDataset(Dataset):
    def __init__(self, X, y, classes, transform=None):
        self.X = X
        self.y = y
        self.transform = transform
        self.classes = classes

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        X = self.X[idx]
        y = self.y[idx]

        if self.transform:
            X = torch.from_numpy(self.transform(image=X.numpy().transpose(1, 2, 0))['image'].transpose(2, 0, 1))

        return X, y


def split_dataset(images_np, labels_np, test_size, random_state):


    return X_train_np, y_train_np, X_val_np, y_val_np, mean_train, std_train
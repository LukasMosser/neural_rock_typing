import torch
from torch.utils.data import Dataset
from imageio import imread
import numpy as np
from neural_rock import preprocess as pre
from sklearn.model_selection import StratifiedShuffleSplit
from skimage.util import img_as_float

class ThinSectionDataset(Dataset):
    def __init__(self, path, label_set, transform=None, seed=42, test_split_size=0.5, train=True, preload_images=True):
        super(ThinSectionDataset, self).__init__()
        self.path = path
        self.train = train
        self.label_set = label_set
        self.preload_images = preload_images
        self.transform = transform
        self.seed = seed
        self.test_split_size = test_split_size
        self.dataset, self.images, self.class_names, self.modified_label_map, self.weights = self._make_dataset()
        self.image_ids = list(self.dataset.keys())

    def _make_dataset(self):
        available_images = pre.load_images(self.path)
        if self.path.split("/")[-1] == "Leg194":
            db_id = "Leg194"
        elif self.path.split("/")[-1] == "Buhasa":
            db_id = "B"
        elif self.path.split("/")[-1] == "Malampaya":
            db_id = "M"

        df = pre.load_excel(db_id)

        available_images, df_ = pre.load_features(available_images, df)

        available_images = {key: values for key, values in available_images.items() if len(values) == 8}

        if self.label_set == "Lucia":
            modified_label_map = {'0': 0, '1': 1, '2': 2}
            class_names = list(modified_label_map.keys())
        elif self.label_set == "DominantPore":
            modified_label_map = {'IP': 0, 'VUG': 1, 'MO': 2, 'IX': 3, 'WF': 4, 'WP': 4}
            class_names = ['IP', 'VUG', 'MO', 'IX', 'WF-WP']
        else:
            modified_label_map = {'rDol': 0, 'B': 1, 'FL': 2,
                                  'G': 3, 'G-P': 4, 'P': 5, 'P-G': 4}
            class_names = ['rDol', 'B', 'FL', 'G', 'G-P,P-G', 'P']

        for idx, dset in available_images.items():
            dset['feature'] = modified_label_map[dset[self.label_set]]

        pore_type = np.array(
            [modified_label_map[str(features[self.label_set])] for idx, features in available_images.items()])
        imgs = np.array(list(available_images.keys()))
        splitter = StratifiedShuffleSplit(test_size=self.test_split_size, random_state=self.seed)

        for train_idx, val_idx in splitter.split(imgs, pore_type):
            break

        imgs_train, pore_type_train = imgs[train_idx], pore_type[train_idx]
        imgs_val, pore_type_val = imgs[val_idx], pore_type[val_idx]

        _, class_frequency = np.unique(pore_type_train, return_counts=True)
        class_frequency = class_frequency / np.sum(class_frequency)

        weights = torch.from_numpy(1. / class_frequency)

        train_dset = {key: data for key, data in available_images.items() if key in imgs_train}
        val_dset = {key: data for key, data in available_images.items() if key in imgs_val}

        dataset = train_dset

        if not self.train:
            dataset = val_dset

        images = {}
        if self.preload_images:
            for key, dset in dataset.items():
               img = imread(dset['path'])
               mask = (1-(imread(dset['mask_path']) > 0).astype(np.uint8))
               img = img * np.repeat(np.expand_dims(mask, axis=-1), 3, axis=-1)
               images[key] = img

        return dataset, images, class_names, modified_label_map, weights

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        dset = self.dataset[self.image_ids[idx]]

        X = torch.from_numpy(img_as_float(self.images[self.image_ids[idx]])).permute(2, 0, 1).float()

        y = torch.from_numpy(np.array([dset['feature']]))

        if self.transform:
            X = self.transform(X)
        return X.float(), y


class GPUThinSectionDataset(Dataset):
    def __init__(self, path, label_set, transform=None, seed=42, test_split_size=0.5, train=True, preload_images=True):
        super(GPUThinSectionDataset, self).__init__()
        self.path = path
        self.train = train
        self.label_set = label_set
        self.preload_images = preload_images
        self.transform = transform
        self.seed = seed
        self.test_split_size = test_split_size
        self.dataset, self.images, self.class_names, self.modified_label_map, self.weights = self._make_dataset()
        self.image_ids = list(self.dataset.keys())
        self.images, self.labels = self.make_dataset()
        self.num_classes = len(self.class_names)

    def _make_dataset(self):
        available_images = pre.load_images(self.path)

        if self.path.split("/")[-1] == "Leg194":
            db_id = "Leg194"

        df = pre.load_excel(db_id)

        available_images, df_ = pre.load_features(available_images, df)

        available_images = {key: values for key, values in available_images.items() if len(values) == 8}

        if self.label_set == "Lucia":
            modified_label_map = {'0': 0, '1': 1, '2': 2}
            class_names = list(modified_label_map.keys())
        elif self.label_set == "DominantPore":
            modified_label_map = {'IP': 0, 'VUG': 1, 'MO': 2, 'IX': 3, 'WF': 4, 'WP': 4}
            class_names = ['IP', 'VUG', 'MO', 'IX', 'WF-WP']
        else:
            modified_label_map = {'rDol': 0, 'B': 1, 'FL': 2,
                                  'G': 3, 'G-P': 4, 'P': 5, 'P-G': 4}
            class_names = ['rDol', 'B', 'FL', 'G', 'G-P,P-G', 'P']

        for idx, dset in available_images.items():
            dset['feature'] = modified_label_map[dset[self.label_set]]

        pore_type = np.array(
            [modified_label_map[str(features[self.label_set])] for idx, features in available_images.items()])
        imgs = np.array(list(available_images.keys()))
        splitter = StratifiedShuffleSplit(test_size=self.test_split_size, random_state=self.seed)

        for train_idx, val_idx in splitter.split(imgs, pore_type):
            break

        imgs_train, pore_type_train = imgs[train_idx], pore_type[train_idx]
        imgs_val, pore_type_val = imgs[val_idx], pore_type[val_idx]

        _, class_frequency = np.unique(pore_type_train, return_counts=True)
        class_frequency = class_frequency / np.sum(class_frequency)

        weights = torch.from_numpy(1. / class_frequency)

        train_dset = {key: data for key, data in available_images.items() if key in imgs_train}
        val_dset = {key: data for key, data in available_images.items() if key in imgs_val}

        dataset = train_dset

        if not self.train:
            dataset = val_dset

        images = {}
        if self.preload_images:
            for key, dset in dataset.items():
                img = imread(dset['path'])
                mask = (1 - (imread(dset['mask_path']) > 0).astype(np.uint8))
                img = img * np.repeat(np.expand_dims(mask, axis=-1), 3, axis=-1)
                images[key] = img

        return dataset, images, class_names, modified_label_map, weights

    def make_dataset(self):
        imgs = [torch.from_numpy(img_as_float(self.images[self.image_ids[idx]])).permute(2, 0, 1).cuda().half() for idx in range(len(self.images))]
        labels = [torch.tensor(self.dataset[self.image_ids[idx]]['feature'], device='cuda') for idx in range(len(self.images))]
        return imgs, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        X = self.images[idx]

        y = self.labels[idx]

        if self.transform:
            X = self.transform(X)

        return X.float(), y
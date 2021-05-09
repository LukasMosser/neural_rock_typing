import torch
from torch.utils.data import Dataset
from imageio import imread
import numpy as np
from neural_rock import preprocess as pre
from sklearn.model_selection import StratifiedShuffleSplit
from skimage.util import img_as_float
from neural_rock.preprocess import load_label_dataframe
from neural_rock.data_models import make_image_dataset, LabelSet, PreloadedImageDataset, GPUPreloadedImageDataset


class ThinSectionDataset(Dataset):
    def __init__(self, base_path, label_set, transform=None, seed=42, test_split_size=0.5, train=True, preload_images=True):
        super(ThinSectionDataset, self).__init__()
        self.base_path = base_path
        self.train = train
        self.label_set = label_set
        self.preload_images = preload_images
        self.transform = transform
        self.seed = seed
        self.test_split_size = test_split_size
        self.dataset, self.labels = self._make_dataset()
        self.num_classes = len(self.labels.class_names)

        if self.preload_images:
            self.dataset = self._preload_images()

    def _make_dataset(self):
        df = load_label_dataframe(self.base_path)
        image_dataset = make_image_dataset(df, base_path=self.base_path)

        splitter = StratifiedShuffleSplit(test_size=self.test_split_size, random_state=self.seed)

        dominant_pore = LabelSet(label_map={'IP': 0, 'VUG': 1, 'MO': 2, 'IX': 3, 'WF': 4, 'WP': 4, 'WF-WP': 4},
                                 class_names=['IP', 'VUG', 'MO', 'IX', 'WF-WP'],
                                 sample_labels={idx: label for idx, label in
                                                df[['Sample', 'Macro_Dominant_type']].values})

        dunham = LabelSet(label_map={'rDol': 0, 'B': 1, 'FL': 2, 'G': 3, 'G-P': 4, 'P': 5, 'P-G': 4, 'G-P,P-G': 4},
                          class_names=['rDol', 'B', 'FL', 'G', 'G-P,P-G', 'P'],
                          sample_labels={idx: label for idx, label in df[['Sample', 'Dunham_class']].values})

        lucia = LabelSet(label_map={'0': 0, '1': 1, '2': 2},
                         class_names=['0', '1', '2'],
                         sample_labels={idx: label for idx, label in df[['Sample', 'Lucia_class']].values})

        if self.label_set == "Dunham":
            label_set = dunham
        elif self.label_set == "DominantPore":
            label_set = dominant_pore
        elif self.label_set == "Lucia":
            label_set = lucia

        imgs = np.array(label_set.samples)
        labels = np.array(label_set.labels)

        for train_idx, val_idx in splitter.split(imgs, labels):
            break

        if self.train:
            dataset = image_dataset.subset(list(imgs[train_idx]))
        else:
            dataset = image_dataset.subset(list(imgs[val_idx]))

        return dataset, label_set

    def _preload_images(self):
        images = {}
        if self.preload_images:
            for key in self.dataset.image_paths.keys():
                img = imread(self.dataset.image_paths[key])
                mask = (1 - (imread(self.dataset.roi_paths[key]) > 0).astype(np.uint8))
                img = img * np.repeat(np.expand_dims(mask, axis=-1), 3, axis=-1)
                images[key] = img

            dataset = PreloadedImageDataset(image_paths=self.dataset.image_paths,
                                            roi_paths=self.dataset.roi_paths,
                                            images=images)
        return dataset

    def __len__(self):
        return len(self.dataset.image_paths)

    def __getitem__(self, idx):
        sample = list(self.dataset.image_paths.keys())[idx]

        X = torch.from_numpy(img_as_float(self.dataset.images[sample])).permute(2, 0, 1)

        y = torch.tensor(self.labels.label_to_class(self.labels.sample_labels[sample]))

        if self.transform:
            X = self.transform(X)
        return X.float(), y


class GPUThinSectionDataset(ThinSectionDataset):
    def __init__(self, *args, **kwargs):
        super(GPUThinSectionDataset, self).__init__(*args, **kwargs)
        self.dataset = self._gpu_preload_images()

    def _gpu_preload_images(self):
        for idx, img in self.dataset.images:
            self.dataset.images[idx] = torch.from_numpy(img_as_float(img)).permute(2, 0, 1).cuda().half()

        sample_labels = self.labels.get_sample_labels_as_class_idx()
        labels = {idx: torch.tensor(label, device='cuda') for idx, label in sample_labels.items()}

        dataset = GPUPreloadedImageDataset(image_paths=self.dataset.image_paths,
                                           roi_paths=self.dataset.roi_paths,
                                           images=self.dataset.images,
                                           labels=labels)
        return dataset

    def __len__(self):
        return len(self.dataset.images.keys())

    def __getitem__(self, idx):
        sample = list(self.dataset.images.keys())[idx]
        X = self.images[sample].float()

        y = self.labels[sample]

        if self.transform:
            X = self.transform(X)

        return X.float(), y
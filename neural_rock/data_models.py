from enum import Enum

import pandas as pd
import pydantic
from typing import List, Dict, Tuple
from pathlib import Path
import torch
import numpy
from torch import nn as nn

from neural_rock.model import make_resnet18_model, make_vgg11_model
from neural_rock.preprocess import get_image_paths


class ImageDataset(pydantic.BaseModel):
    image_paths: Dict[int, pydantic.FilePath]
    roi_paths: Dict[int, pydantic.FilePath]

    def subset(self, samples):
        return ImageDataset(image_paths={k: self.image_paths[k] for k in samples},
                            roi_paths={k: self.roi_paths[k] for k in samples})


class PreloadedImageDataset(ImageDataset):
    images: Dict[int, numpy.ndarray]

    def __init_subclass__(cls, optional_fields=None, **kwargs):
        super().__init_subclass__(**kwargs)

    class Config:
        arbitrary_types_allowed = True


class GPUPreloadedImageDataset(ImageDataset):
    images: Dict[int, torch.Tensor]
    labels: Dict[int, torch.Tensor]

    def __init_subclass__(cls, optional_fields=None, **kwargs):
        super().__init_subclass__(**kwargs)

    class Config:
        arbitrary_types_allowed = True


class LabelSet(pydantic.BaseModel):
    label_map: Dict[str, int]
    class_names: List[str]
    sample_labels: Dict[int, str]

    def label_to_class(self, label):
        return self.label_map[label]

    def class_to_label(self, class_idx):
        return self.class_names[class_idx]

    def get_sample_labels_as_class_idx(self):
        return {idx: self.label_to_class(val) for idx, val in self.sample_labels.items()}

    @property
    def labels(self):
        return [val for _, val in self.sample_labels.items()]

    @property
    def samples(self):
        return list(self.sample_labels.keys())

    def subset(self, samples):
        return LabelSet(label_map=self.label_map,
                        class_names=self.class_names,
                        sample_labels=self.sample_labels[samples])

    @property
    def num_classes(self):
        return len(self.class_names)


def make_image_dataset(df: pd.DataFrame, base_path: Path=Path("..")) -> ImageDataset:
    sample_ids, image_paths, roi_paths = get_image_paths(base_path=base_path)
    image_paths = {idx: path for idx, path in image_paths.items() if idx in df['Sample'].values}
    roi_paths = {idx: path for idx, path in roi_paths.items() if idx in df['Sample'].values}
    image_dataset = ImageDataset(image_paths=image_paths, roi_paths=roi_paths)
    return image_dataset


class ModelName(str, Enum):
    resnet = "resnet"
    vgg = "vgg"


class LabelSetName(str, Enum):
    dunham = "Dunham"
    lucia = "Lucia"
    dominant_pore = "DominantPore"


class LabelSets(pydantic.BaseModel):
    sets: Dict[LabelSetName, LabelSet]


class Model(pydantic.BaseModel):
    label_set: LabelSetName
    model_name: ModelName
    frozen: bool
    path: str
    train_test_split: Dict[str, List[int]]

    def get_model(self, num_classes) -> Tuple[nn.Module]:
        if self.model_name == 'resnet':
            return make_resnet18_model(num_classes=num_classes, pretrained=False)
        else:
            return make_vgg11_model(num_classes=num_classes, pretrained=False)


class ModelZoo(pydantic.BaseModel):
    models: List[Model]


class CAMRequest(pydantic.BaseModel):
    cam_map: List[List[float]]
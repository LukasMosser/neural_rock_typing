from enum import Enum
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
import pydantic
import torch
from torch import nn as nn
import numpy
from neural_rock.model import make_resnet18_model, make_vgg11_model
from neural_rock.preprocess import get_image_paths


class ImageDataset(pydantic.BaseModel):
    """
    Contains paths to the images and the ROI images that mask the outer regions of the thin-sections.
    """
    image_paths: Dict[int, pydantic.FilePath]
    roi_paths: Dict[int, pydantic.FilePath]

    def subset(self, samples):
        return ImageDataset(image_paths={k: self.image_paths[k] for k in samples},
                            roi_paths={k: self.roi_paths[k] for k in samples})


class PreloadedImageDataset(ImageDataset):
    """
    A container class for preloaded Images. Deprecated
    """
    images: Dict[int, numpy.ndarray]

    def __init_subclass__(cls, optional_fields=None, **kwargs):
        super().__init_subclass__(**kwargs)

    class Config:
        arbitrary_types_allowed = True


class GPUPreloadedImageDataset(ImageDataset):
    """
    Data class to store preloaded Tensors.
    Used for training on Colab.
    """
    images: Dict[int, torch.Tensor]
    labels: Dict[int, torch.Tensor]

    def __init_subclass__(cls, optional_fields=None, **kwargs):
        super().__init_subclass__(**kwargs)

    class Config:
        arbitrary_types_allowed = True


class LabelSet(pydantic.BaseModel):
    """
    Base data model for Labelsets.
    Provide access to each label sets class names, a mapping between index and class label representation
    and assignment of each samples label.
    """
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


class ModelName(str, Enum):
    """
    Enum for valid model names defined in Neural Rock
    """
    resnet = "resnet"
    vgg = "vgg"


class LabelSetName(str, Enum):
    """
    Enum for valid Labelset names in Neural Rock
    """
    dunham = "Dunham"
    lucia = "Lucia"
    dominant_pore = "DominantPore"


class LabelSets(pydantic.BaseModel):
    """
    Base model for label sets. Used in the API of Neural Rock
    """
    sets: Dict[LabelSetName, LabelSet]


class Model(pydantic.BaseModel):
    """
    Defines a Model and it's configuration.
    Used for the api and to grab info about each of the provided models.
    """
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
    """
    Contains all the models defined in Neural Rock
    """
    models: List[Model]


class CAMRequest(pydantic.BaseModel):
    """
    Return type for the API used to provide the cam_map
    """
    cam_map: List[List[float]]


def make_image_dataset(df: pd.DataFrame,
                       base_path: Path=Path("..")) -> ImageDataset:
    """
    Creates the image dataset for the Leg194 dataset.
    """
    sample_ids, image_paths, roi_paths = get_image_paths(base_path=base_path)
    image_paths = {idx: path for idx, path in image_paths.items() if idx in df['Sample'].values}
    roi_paths = {idx: path for idx, path in roi_paths.items() if idx in df['Sample'].values}
    image_dataset = ImageDataset(image_paths=image_paths, roi_paths=roi_paths)
    return image_dataset
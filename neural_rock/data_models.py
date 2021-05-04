import pandas as pd
import pydantic
from typing import List, Dict
from pathlib import Path

import torch

from neural_rock.preprocess import get_image_paths
import numpy


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


def make_image_dataset(df: pd.DataFrame, base_path: Path=Path("..")):
    sample_ids, image_paths, roi_paths = get_image_paths(base_path=base_path)
    image_paths = {idx: path for idx, path in image_paths.items() if idx in df['Sample'].values}
    roi_paths = {idx: path for idx, path in roi_paths.items() if idx in df['Sample'].values}
    image_dataset = ImageDataset(image_paths=image_paths, roi_paths=roi_paths)
    return image_dataset
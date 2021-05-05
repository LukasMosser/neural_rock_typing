from pathlib import Path

import numpy as np
from imageio import imread

from neural_rock.data_models import LabelSet, LabelSets, Model, ModelZoo, LabelSetName, ModelName, ImageDataset
from neural_rock.utils import get_train_test_split


def make_label_sets(df):
    dominant_pore = LabelSet(label_map={'IP': 0, 'VUG': 1, 'MO': 2, 'IX': 3, 'WF': 4, 'WP': 4},
                             class_names=['IP', 'VUG', 'MO', 'IX', 'WF-WP'],
                             sample_labels={idx: label for idx, label in
                                            df[['Sample', 'Macro_Dominant_type']].values})

    dunham = LabelSet(label_map={'rDol': 0, 'B': 1, 'FL': 2, 'G': 3, 'G-P': 4, 'P': 5, 'P-G': 4},
                      class_names=['rDol', 'B', 'FL', 'G', 'G-P,P-G', 'P'],
                      sample_labels={idx: label for idx, label in df[['Sample', 'Dunham_class']].values})

    lucia = LabelSet(label_map={'0': 0, '1': 1, '2': 2},
                     class_names=['0', '1', '2'],
                     sample_labels={idx: label for idx, label in df[['Sample', 'Lucia_class']].values})
    return LabelSets(sets={'DominantPore': dominant_pore,
                                 'Dunham': dunham,
                                 'Lucia': lucia})


def init_model_zoo(base_path: Path):
    models = []
    for label_set in ['Dunham', 'DominantPore', 'Lucia']:
        for model in ['vgg', 'resnet']:
            for frozen in [True, False]:
                path = Path('data/models/{0:}/{1:}/{2:}/'.format(label_set, model, str(frozen)))
                checkpoint_path = base_path.joinpath(path, 'best.ckpt')
                train_test_path = base_path.joinpath(path, 'train_test_split.json')
                train_test_split = get_train_test_split(path=train_test_path)

                models.append(Model(label_set=label_set,
                                    model_name=model,
                                    frozen=frozen,
                                    path=str(checkpoint_path),
                                    train_test_split=train_test_split))
    return ModelZoo(models=models)


def model_lookup(model_zoo: ModelZoo, label_set: LabelSetName, model_name: ModelName, frozen: bool):
    for model in model_zoo.models:
        if model.label_set == label_set and model.model_name == model_name and model.frozen == frozen:
            return model
    else:
        return None


def load_image(image_dataset: ImageDataset, sample_id: int):
    img = imread(image_dataset.image_paths[sample_id])
    mask = imread(image_dataset.roi_paths[sample_id])
    mask = (1 - (mask > 0).astype(np.uint8))
    img = img * np.repeat(np.expand_dims(mask, axis=-1), 3, axis=-1)
    img = np.transpose(img, (1, 0, 2))
    return img
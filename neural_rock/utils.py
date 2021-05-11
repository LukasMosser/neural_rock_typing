import json


MEAN_TRAIN = [0.485, 0.456, 0.406]
STD_TRAIN = [0.229, 0.224, 0.225]


def get_train_test_split(path="./data/train_test_split.json"):
    """
    Loads the train test split indicators for a training run.
    """
    with open(path) as f:
        train_test_split = json.load(f)
    return train_test_split
from fastapi import FastAPI
from mangum import Mangum
from pathlib import Path
from typing import List, Dict
import torch.nn as nn
import numpy as np
from neural_rock.app.utils import make_cam_map
from neural_rock.model import NeuralRockModel
from neural_rock.cam import GradCam
from neural_rock.server.utils import make_label_sets, init_model_zoo, model_lookup, load_image
from neural_rock.data_models import make_image_dataset, ModelName, LabelSetName, LabelSets, CAMRequest
from neural_rock.preprocess import load_label_dataframe

app = FastAPI(title='Neural Rock API')

device = 'cpu'
valid_layers = {'resnet': list(range(8)), 'vgg': list(range(21))}
base_path = Path("..")

df = load_label_dataframe(base_path=base_path)
image_dataset = make_image_dataset(df, base_path=base_path)
label_sets = make_label_sets(df)
model_zoo = init_model_zoo(base_path=base_path)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/models")
async def models():
    return model_zoo


@app.get("/dataset")
async def dataset():
    return image_dataset


@app.get("/labelsets")
async def labelsets() -> LabelSets:
    return label_sets


@app.get("/dataset/img/{sample_id}")
async def get_img_path(sample_id: int) -> Path:
    return image_dataset.image_paths[sample_id]


@app.get("/dataset/roi/{sample_id}")
async def get_roi_path(sample_id: int) -> Path:
    return image_dataset.roi_paths[sample_id]


@app.get("/dataset/sample_ids")
async def get_sample_ids() -> List[int]:
    return list(image_dataset.image_paths.keys())

@app.get("/models/{model_name}/valid_layer_range")
async def get_valid_layer_range(model_name: ModelName) -> List[int]:
    return valid_layers[model_name]


@app.get("/get_train_test_images/{label_set}/{model_name}/{frozen}")
async def get_train_test_images(label_set: LabelSetName,
                              model_name: ModelName,
                              frozen: bool) -> Dict[str, List[int]]:
    model_config = model_lookup(model_zoo, label_set, model_name, frozen)
    return model_config.train_test_split


@app.get("/cam/{label_set}/{model_name}/{layer_id}/{frozen}/{sample_id}/{class_name}")
async def compute_cam(label_set: LabelSetName,
                      model_name: ModelName,
                      layer_id: int,
                      frozen: bool,
                      sample_id: int,
                      class_name: str) -> CAMRequest:
    print(class_name)
    assert class_name in label_sets.sets[label_set].class_names
    model_config = model_lookup(model_zoo, label_set, model_name, frozen)
    assert layer_id in valid_layers[model_name]

    feature_extractor, classifier = model_config.get_model(label_sets.sets[label_set].num_classes)

    model = NeuralRockModel(feature_extractor=feature_extractor,
                            classifier=classifier,
                            num_classes=label_sets.sets[label_set].num_classes)

    model = model.load_from_checkpoint(model_config.path).to(device)
    model.eval()
    model.freeze()

    if model_config.model_name.lower() == 'resnet':
        model.feature_extractor = nn.Sequential(*list(model.feature_extractor.children()))

    cam = GradCam(model=model,
                  feature_module=model.feature_extractor,
                  target_layer_names=[str(layer_id)],
                  device=device)

    img = load_image(image_dataset, sample_id)
    map = make_cam_map(img,
                       cam,
                       label_sets.sets[label_set].label_to_class(class_name),
                       device=device)
    print(map.shape)
    cam_request = CAMRequest(cam_map=np.nan_to_num(map).tolist())
    return cam_request.cam_map

handler = Mangum(app=app)

#if __name__ == "__main__":
#    uvicorn.run(app, host="0.0.0.0", port=8000)

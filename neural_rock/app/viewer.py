import param
from holoviews.operation.datashader import rasterize
import numpy as np
from neural_rock.utils import get_train_test_split
from neural_rock.app.plot import create_holoviews_thinsection, create_holoviews_cam
from imageio import imread
import holoviews as hv
from neural_rock.data_models import ImageDataset, ModelZoo, LabelSets
from neural_rock.server.utils import model_lookup
from torchvision import transforms
from pathlib import Path
import torch
import requests
from json import loads
hv.extension('bokeh')


class ThinSectionViewer(param.Parameterized):
    label_set_name = param.ObjectSelector(default="Lucia", objects=["Dunham", 'Lucia', 'DominantPore'])

    model_selector = param.ObjectSelector(default="resnet", objects=['vgg', 'resnet'])
    frozen_selector = param.ObjectSelector(default="True", objects=['True', 'False'])

    Network_Layer_Number = param.Integer(default=0, bounds=(0, 1))

    Class_Name = param.ObjectSelector(default="default", objects=["default"])

    Image_Name = param.ObjectSelector(default="default", objects=["default"])

    def __init__(self, server_address: Path,
                 image_dataset: ImageDataset, label_sets: LabelSets, model_zoo: ModelZoo,
                 **params):
        super(ThinSectionViewer, self).__init__(**params)
        self.server_address = server_address
        self.image_dataset = image_dataset
        self.label_sets = label_sets
        self.model_zoo = self._get_model_zoo()
        self.label_sets = self._get_labelsets()
        self._get_layer_ranges()
        self.update_class_names()
        self._get_available_image_ids()

    def _get_model_zoo(self) -> ModelZoo:
        with requests.Session() as s:
            result = s.get(self.server_address + 'models')
            r = loads(result.text)
            model_zoo = ModelZoo(**r)
        return model_zoo

    def _get_labelsets(self) -> LabelSets:
        with requests.Session() as s:
            result = s.get(self.server_address + 'labelsets')
            r = loads(result.text)
            labelsets = LabelSets(**r)

        self.param['label_set_name'].default = list(labelsets.sets.keys())[0]
        self.param['label_set_name'].objects = list(labelsets.sets.keys())

        return labelsets

    @param.depends('label_set_name', watch=True)
    def update_class_names(self):
        class_names = self.label_sets.sets[self.label_set_name].class_names
        self.param['Class_Name'].default = class_names[0]
        self.param['Class_Name'].objects = class_names

    def _get_available_image_ids(self):
        with requests.Session() as s:
            result = s.get(self.server_address + 'dataset/sample_ids')
            r = loads(result.text)

        train_test_split = self._get_train_test_config()

        samples_text_map = {}
        for sample_id in r:
            if sample_id in train_test_split['train']:
                samples_text_map["{0:}-Train".format(sample_id)] = sample_id
            elif sample_id in train_test_split['test']:
                samples_text_map["{0:}-Test".format(sample_id)] = sample_id

        self.samples_text_map = samples_text_map
        self.param['Image_Name'].default = self.samples_text_map
        self.param['Image_Name'].objects = self.samples_text_map

    @param.depends('label_set_name', 'model_selector', 'frozen_selector', watch=True)
    def _get_train_test_config(self):
        with requests.Session() as s:
            result = s.get(self.server_address + 'get_train_test_images/{0:}/{1:}/{2:}'.format(self.label_set_name,
                                                                                               self.model_selector.lower(),
                                                                                               self.frozen_selector))
            r = loads(result.text)

        return r

    @param.depends('label_set_name', 'model_selector', 'frozen_selector', watch=True)
    def _load_train_test_split(self):
        checkpoint_path = "./data/models/{0:}/{1:}/{2:}/".format(self.label_set_name,
                                                                 self.model_selector.lower(),
                                                                 self.frozen_selector)

        self.train_test_split = get_train_test_split(path=checkpoint_path + "train_test_split.json")

    @param.depends('model_selector', watch=True)
    def _get_layer_ranges(self):
        with requests.Session() as s:
            result = s.get(self.server_address + 'models/{0:}/valid_layer_range'.format(self.model_selector))
            r = loads(result.text)

        min_layer = r[0]
        max_layer = r[-1]
        self.param['Network_Layer_Number'].default = max_layer
        self.param['Network_Layer_Number'].bounds = (min_layer, max_layer)
        self.param['Network_Layer_Number'].default = max_layer

    @param.depends('label_set_name', 'model_selector', 'frozen_selector', 'Network_Layer_Number', 'Class_Name', 'Image_Name')
    def load_image(self, image_name):
        image_id = self.samples_text_map[image_name]
        try:
            X_np = imread(self.image_dataset.image_paths[image_id])
            mask = imread(self.image_dataset.roi_paths[image_id])
            mask = (1 - (mask > 0).astype(np.uint8))
            X_np = X_np * np.repeat(np.expand_dims(mask, axis=-1), 3, axis=-1)
            X_np = np.transpose(X_np, (1, 0, 2))
        except KeyError:
            X_np = np.random.uniform(0, 1, size=(1000, 1000, 3))
        return X_np

    @param.depends('label_set_name', 'model_selector', 'frozen_selector', 'Network_Layer_Number', 'Class_Name', 'Image_Name')
    def make_map(self):
        sample_id = self.samples_text_map[self.Image_Name]
        with requests.Session() as s:
            result = s.get(self.server_address + 'cam/{0:}/{1:}/{2:}/{3:}/{4:}/{5:}'.format(self.label_set_name,
                                                                                  self.model_selector.lower(),
                                                                                  self.Network_Layer_Number,
                                                                                  self.frozen_selector,
                                                                                  sample_id,
                                                                                  self.Class_Name))
            r = loads(result.text)
            map = np.array(r)
        return map

    @param.depends('label_set_name', 'model_selector', 'frozen_selector', 'Network_Layer_Number', 'Class_Name', 'Image_Name')
    def load_symbol(self):
        X_np = self.load_image(image_name=self.Image_Name)
        resize = transforms.Resize((X_np.shape[0], X_np.shape[1]))
        map = self.make_map()
        map = resize(torch.from_numpy(map).unsqueeze(0)).numpy()[0]
        hv_thinsection = create_holoviews_thinsection(X_np[::-1])
        hv_cam = create_holoviews_cam(map.T).opts(alpha=0.5,
                                                             cmap='inferno')
        return hv_thinsection * hv_cam

    def view(self):
        thin_section = hv.DynamicMap(self.load_symbol)

        thin_section = rasterize(thin_section).opts(data_aspect=1.0, frame_height=600, frame_width=1000,
                                                    active_tools=['xwheel_zoom', 'pan'])
        return thin_section
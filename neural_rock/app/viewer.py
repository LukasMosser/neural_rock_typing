import param
from holoviews.operation.datashader import rasterize
import numpy as np
from neural_rock.app.plot import create_holoviews_thinsection, create_holoviews_cam
from imageio import imread
import holoviews as hv
from neural_rock.data_models import ImageDataset, ModelZoo, LabelSets
from torchvision import transforms
from pathlib import Path
import torch
import requests
from json import loads
hv.extension('bokeh')


class ThinSectionViewer(param.Parameterized):
    Labelset_Name = param.ObjectSelector(default="Lucia", objects=["Dunham", 'Lucia', 'DominantPore'], check_on_set=True)

    Model_Selector = param.ObjectSelector(default="resnet", objects=['vgg', 'resnet'], check_on_set=True)
    Frozen_Selector = param.ObjectSelector(default="True", objects=['True', 'False'], check_on_set=True)

    Network_Layer_Number = param.Integer(default=0, bounds=(0, 1))

    Class_Name = param.ObjectSelector(default="default", objects=["default"], check_on_set=True)

    Image_Name = param.ObjectSelector(default="default", objects=["default"], check_on_set=True)

    def __init__(self, server_address: Path,
                 image_dataset: ImageDataset,
                 **params):
        super(ThinSectionViewer, self).__init__(**params)
        self.server_address = server_address
        self.image_dataset = image_dataset
        self.model_zoo = self._get_model_zoo()
        self.label_sets = self._get_labelsets()
        self._get_layer_ranges()
        self._update_class_names()
        self._get_available_image_ids()
        self.map = self.load_symbol

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

        self.param['Labelset_Name'].default = list(labelsets.sets.keys())[0]
        self.param['Labelset_Name'].objects = list(labelsets.sets.keys())
        self.Labelset_Name = list(labelsets.sets.keys())[0]

        return labelsets

    @param.depends('Labelset_Name', watch=True)
    def _update_class_names(self):
        class_names = self.label_sets.sets[self.Labelset_Name].class_names
        self.param['Class_Name'].default = class_names[0]
        self.param['Class_Name'].objects = class_names
        self.Class_Name = class_names[0]

    @param.depends('Labelset_Name', 'Model_Selector')
    def _get_available_image_ids(self):
        with requests.Session() as s:
            result = s.get(self.server_address + 'dataset/sample_ids')
            r = loads(result.text)

        train_test_split = self._get_train_test_config()

        self.label_sets = self._get_labelsets()
        self._update_class_names()

        samples_text_map = {}
        for sample_id in r:
            if sample_id in train_test_split['train']:
                samples_text_map["{0:}-Train".format(sample_id)] = sample_id
            elif sample_id in train_test_split['test']:
                samples_text_map["{0:}-Test".format(sample_id)] = sample_id

        self.samples_text_map = samples_text_map
        self.param['Image_Name'].default = list(self.samples_text_map.keys())[0]
        self.param['Image_Name'].objects = list(self.samples_text_map.keys())
        self.Image_Name = self.param['Image_Name'].objects[0]

    @param.depends('Labelset_Name', 'Model_Selector', 'Frozen_Selector')
    def _get_train_test_config(self):
        with requests.Session() as s:
            result = s.get(self.server_address + 'get_train_test_images/{0:}/{1:}/{2:}'.format(self.Labelset_Name,
                                                                                               self.Model_Selector,
                                                                                               self.Frozen_Selector))
            r = loads(result.text)

        return r

    @param.depends('Model_Selector')
    def _get_layer_ranges(self):
        with requests.Session() as s:
            result = s.get(self.server_address + 'models/{0:}/valid_layer_range'.format(self.Model_Selector))
            r = loads(result.text)

        min_layer = r[0]
        max_layer = r[-1]
        self.param['Network_Layer_Number'].default = max_layer
        self.param['Network_Layer_Number'].bounds = (min_layer, max_layer)
        self.Network_Layer_Number = self.param['Network_Layer_Number'].default

    @param.depends('Image_Name')
    def load_image(self):
        image_id = self.samples_text_map[self.Image_Name]
        try:
            X_np = imread(self.image_dataset.image_paths[image_id])
            mask = imread(self.image_dataset.roi_paths[image_id])
            mask = (1 - (mask > 0).astype(np.uint8))
            X_np = X_np * np.repeat(np.expand_dims(mask, axis=-1), 3, axis=-1)
            X_np = np.transpose(X_np, (1, 0, 2))
        except KeyError:
            X_np = np.random.uniform(0, 1, size=(1000, 1000, 3))
        return X_np

    @param.depends('Image_Name', 'Model_Selector', 'Labelset_Name', 'Frozen_Selector', 'Class_Name')
    def make_map(self):
        print(self.Class_Name, self.param['Class_Name'].objects)
        sample_id = self.samples_text_map[self.Image_Name]
        with requests.Session() as s:
            result = s.get(self.server_address + 'cam/{0:}/{1:}/{2:}/{3:}/{4:}/{5:}'.format(self.Labelset_Name,
                                                                                              self.Model_Selector,
                                                                                              self.Network_Layer_Number,
                                                                                              self.Frozen_Selector,
                                                                                              sample_id,
                                                                                              self.Class_Name))
            r = loads(result.text)
            map = np.array(r)
        return map

    @param.depends('Image_Name')
    def load_symbol(self):
        X_np = self.load_image()
        resize = transforms.Resize((X_np.shape[0], X_np.shape[1]))
        map = self.make_map()
        map = resize(torch.from_numpy(map).unsqueeze(0)).numpy()[0]
        hv_thinsection = create_holoviews_thinsection(X_np[::-1])
        hv_cam = create_holoviews_cam(map.T).opts(alpha=0.5, cmap='inferno')
        return hv_thinsection * hv_cam

    def view(self):
        thin_section = hv.DynamicMap(self.map)

        thin_section = rasterize(thin_section).opts(data_aspect=1.0, frame_height=600, frame_width=1000,
                                                    active_tools=['xwheel_zoom', 'pan'])
        return thin_section
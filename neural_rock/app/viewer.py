import param
from holoviews.operation.datashader import rasterize
import numpy as np
from neural_rock.cam import GradCam, make_maps
from neural_rock.app.utils import load_model, load_data, get_train_test_split, Model
from neural_rock.app.plot import create_holoviews_thinsection, create_holoviews_cam
from imageio import imread
from neural_rock.model import make_vgg11_model, make_resnet18_model
import torch.nn as nn
import holoviews as hv
hv.extension('bokeh')


class ThinSectionViewer(param.Parameterized):
    label_set_name = param.ObjectSelector(default="Lucia", objects=["Dunham", 'Lucia', 'DominantPore'])
    device = param.ObjectSelector(default="cpu", objects=["cpu", 'cuda'])
    model_selector = param.ObjectSelector(default="ResNet", objects=['VGG', 'ResNet'])
    frozen_selector = param.ObjectSelector(default="True", objects=['True', 'False'])

    Network_Layer_Number = param.Integer(default=0, bounds=(0, 1))

    Class_Name = param.ObjectSelector(default="default", objects=["default"])

    Image_Name = param.ObjectSelector(default="default", objects=["default"])

    def __init__(self, **params):
        super(ThinSectionViewer, self).__init__(**params)

        self._load_train_test_split()
        self._load_images()
        self._load_model()

    @param.depends('label_set_name', 'model_selector', 'frozen_selector', watch=True)
    def _load_train_test_split(self):
        checkpoint_path = "./data/models/{0:}/{1:}/{2:}/".format(self.label_set_name,
                                                                 self.model_selector.lower(),
                                                                 self.frozen_selector)

        self.train_test_split = get_train_test_split(path=checkpoint_path + "train_test_split.json")

    @param.depends('device', 'label_set_name', 'model_selector', 'frozen_selector', watch=True)
    def _load_model(self):
        checkpoint_path = "./data/models/{0:}/{1:}/{2:}/".format(self.label_set_name,
                                                                 self.model_selector.lower(),
                                                                 self.frozen_selector)

        if self.model_selector.lower() == 'vgg':
            model_func = make_vgg11_model
        elif self.model_selector.lower() == 'resnet':
            model_func = make_resnet18_model

        self.model = load_model(checkpoint_path+"/best.ckpt",
                                num_classes=len(self.class_names),
                                model_init_func=model_func)

        if self.model_selector.lower() == 'resnet':
            self.model.feature_extractor = nn.Sequential(*list(self.model.feature_extractor.children()))

        self.model = Model(self.model.feature_extractor, self.model.classifier)
        self.model.to(self.device)

        min_layer = 0
        max_layer = len(list(self.model.feature_extractor.children()))-2

        self.grad_cam = GradCam(model=self.model.to(self.device),
                           feature_module=self.model.feature_extractor.to(self.device),
                           target_layer_names=[str(max_layer)],
                           device=self.device)

        self.param['Network_Layer_Number'].default = max_layer
        self.param['Network_Layer_Number'].bounds = (min_layer, max_layer)
        self.param['Network_Layer_Number'].default = max_layer

    @param.depends('label_set_name', watch=True)
    def _load_images(self):

        labelset = self.label_set_name

        image_paths, modified_label_map, image_names, class_names = load_data(labelset)

        for id, img in image_paths.items():
            if id in self.train_test_split['train']:
                img['train'] = 'Train'
            else:
                img['train'] = 'Test'

        image_name_map = {}
        image_names = []
        for image_name, dset in image_paths.items():
            image_n = "-".join([str(image_name), dset['label'], str(dset['train'])])
            image_names.append(image_n)
            image_name_map[image_n] = image_name

        self.image_name_map = image_name_map
        self.image_names = image_names
        self.class_names = class_names
        self.image_paths = image_paths

        self.param['Class_Name'].default = self.class_names[0]
        self.param['Class_Name'].objects = self.class_names
        self.param['Image_Name'].default = self.image_names[0]
        self.param['Image_Name'].objects = self.image_names

    @param.depends('label_set_name', 'model_selector', 'frozen_selector', 'Network_Layer_Number', 'Class_Name', 'Image_Name')
    def load_image(self, image_id):
        X_np = imread(self.image_paths[image_id]['path'])
        mask = imread(self.image_paths[image_id]['mask_path'])
        mask = (1 - (mask > 0).astype(np.uint8))
        X_np = X_np * np.repeat(np.expand_dims(mask, axis=-1), 3, axis=-1)
        X_np = np.transpose(X_np, (1, 0, 2))
        maps = make_maps(X_np, self.grad_cam, num_classes=len(self.class_names))
        return X_np, maps

    @param.depends('label_set_name', 'model_selector', 'frozen_selector', 'Network_Layer_Number', 'Class_Name', 'Image_Name')
    def make_image_maps(self):
        self.grad_cam.target_layer_names = [str(self.Network_Layer_Number)]
        image_id = self.image_name_map[self.Image_Name]
        X_np, maps = self.load_image(image_id)
        return X_np, maps

    @param.depends('label_set_name', 'model_selector', 'frozen_selector', 'Network_Layer_Number', 'Class_Name', 'Image_Name')
    def load_symbol(self):
        X_np, maps = self.make_image_maps()
        hv_thinsection = create_holoviews_thinsection(X_np[::-1])
        class_idx = np.argwhere(np.array(self.class_names) == self.Class_Name)[0, 0]
        hv_cam = create_holoviews_cam(maps[class_idx].T).opts(alpha=0.5,
                                                              cmap='inferno')
        return hv_thinsection * hv_cam

    def view(self):
        thin_section = hv.DynamicMap(self.load_symbol)

        thin_section = rasterize(thin_section).opts(data_aspect=1.0, frame_height=600, frame_width=1000,
                                                    active_tools=['xwheel_zoom', 'pan'])
        return thin_section
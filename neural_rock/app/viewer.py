from json import loads
from pathlib import Path
import requests
import numpy as np
from imageio import imread
import param
import holoviews as hv
from holoviews.operation.datashader import rasterize
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sn
from neural_rock.data_models import ImageDataset, ModelZoo, LabelSets
import bokeh
from neural_rock.app.plot import create_holoviews_thinsection, create_holoviews_cam
hv.extension('bokeh')

import panel as pn
import logging
import pandas as pd
LOG_FORMAT = "%(levelname)s %(asctime)s - %(message)s"
file_handler = logging.FileHandler(filename='test.log', mode='w')
file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
logger = logging.getLogger(__name__)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)


class ThinSectionViewer(param.Parameterized):
    """
    Base class for the thin-section viewer and the selection widgets.
    Builds on Holoviews Panel and Holoviews Parameters to create parameters that are then turned into Widgets
    throught the front end.
    """
    Labelset_Name = param.ObjectSelector(default="Lucia", objects=["Dunham", 'Lucia', 'DominantPore'], check_on_set=True)

    Model_Selector = param.ObjectSelector(default="resnet", objects=['vgg', 'resnet'], check_on_set=True)
    Frozen_Selector = param.ObjectSelector(default="True", objects=['True', 'False'], check_on_set=True)

    Network_Layer_Number = param.Integer(default=0, bounds=(0, 1))

    Class_Name = param.ObjectSelector(default="default", objects=["default"], check_on_set=True)

    Image_Name = param.ObjectSelector(default="default", objects=["default"], check_on_set=True)
    Show_CAM = param.Boolean(True, doc="Show CAM Map")
    Alpha = param.Number(0.3, bounds=(0.0, 1.0), doc="CAM Alpha")

    def __init__(self, server_address: Path,
                 image_dataset: ImageDataset,
                 **params):
        super(ThinSectionViewer, self).__init__(**params)
        self.cache = []
        self.server_address = server_address
        self.image_dataset = image_dataset
        self.model_zoo = self._get_model_zoo()
        self.label_sets = self._get_labelsets()

        self._get_layer_ranges()
        self._update_class_names()
        self._get_available_image_ids()
        self.map = self.load_symbol
        self.probs = self.make_map()[1]
        self.confusion_matrices = self._get_confusion_matrices(self.Labelset_Name, self.Model_Selector, self.Frozen_Selector)

    def _get_model_zoo(self) -> ModelZoo:
        logger.info("Get Model Zoo")
        """
        Calls API to retrieve available Models in Model Zoo
        """
        with requests.Session() as s:
            result = s.get(self.server_address + 'models')
            r = loads(result.text)
            model_zoo = ModelZoo(**r)
        return model_zoo

    def _get_labelsets(self) -> LabelSets:
        logger.info("Get Labelsets")
        """
        Calls API to retrieve available Labelsets
        """
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
        logger.info("Update Class Names")
        class_names = self.label_sets.sets[self.Labelset_Name].class_names
        self.param['Class_Name'].default = class_names[0]
        self.param['Class_Name'].objects = class_names
        self.Class_Name = class_names[0]

    @param.depends('Labelset_Name', 'Model_Selector', 'Frozen_Selector', watch=True)
    def _get_available_image_ids(self):
        logger.info("Get available image ids")
        """
        Calls API to retrieve available Images for a given Labelset and Model
        """
        with requests.Session() as s:
            result = s.get(self.server_address + 'dataset/sample_ids')
            r = loads(result.text)

        train_test_split = self._get_train_test_config()

        #self.label_sets = self._get_labelsets()
        #self._update_class_names()

        labels = self.label_sets.sets[self.Labelset_Name].sample_labels
        samples_text_map = {}
        for sample_id in r:
            if sample_id in train_test_split['train']:
                samples_text_map["{0:}-Train-True Class-{1:}".format(sample_id, labels[sample_id])] = sample_id
            elif sample_id in train_test_split['test']:
                samples_text_map["{0:}-Test-True Class-{1:}".format(sample_id, labels[sample_id])] = sample_id

        self.samples_text_map = samples_text_map
        self.param['Image_Name'].default = list(self.samples_text_map.keys())[0]
        self.param['Image_Name'].objects = list(self.samples_text_map.keys())
        self.Image_Name = self.param['Image_Name'].objects[0]

    @param.depends('Labelset_Name', 'Model_Selector', 'Frozen_Selector')
    def _get_train_test_config(self):
        logger.info("Get _get_train_test_config")
        """
        Calls API to retrieve the train test split for a specific trained model.
        """
        with requests.Session() as s:
            result = s.get(self.server_address + 'get_train_test_images/{0:}/{1:}/{2:}'.format(self.Labelset_Name,
                                                                                               self.Model_Selector,
                                                                                               self.Frozen_Selector))
            r = loads(result.text)

        return r

    @param.depends('Model_Selector')
    def _get_layer_ranges(self):
        logger.info("Get _get_layer_ranges")
        """
        Calls API to retrieve available layer numbers for the CAM computation
        """
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
        logger.info("Get load_image")
        """
        Loads an image. Shoudld replace with other loading function in future.
        """
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

    def check_cache(self, labelset_Name, Model_Selector, Network_Layer_Number, Frozen_Selector, sample_id, Class_Name):
        for row in self.cache:
            if row['labelset_Name'] == labelset_Name and row['Model_Selector'] == Model_Selector and row['Network_Layer_Number'] == Network_Layer_Number and row['Frozen_Selector'] == Frozen_Selector and row['sample_id'] == sample_id and row['Class_Name'] == Class_Name:
                return row
        return None

    def _get_confusion_matrices(self, labelset_Name, Model_Selector, Frozen_Selector):
        with requests.Session() as s:
            result = s.get(self.server_address + 'confusion_matrices/{0:}/{1:}/{2:}'.format(labelset_Name,
                                                                                            Model_Selector,
                                                                                            Frozen_Selector))
            r = loads(result.text)
            return r['confusion_matrices']

    @param.depends('Image_Name', 'Model_Selector', 'Labelset_Name', 'Frozen_Selector', 'Class_Name')
    def make_map(self):
        logger.info("Get make_map")
        """
        Calls API to generate a cam map for a given image, model, layer number and target class name.
        """
        sample_id = self.samples_text_map[self.Image_Name]

        cache_result = self.check_cache(self.Labelset_Name, self.Model_Selector, self.Network_Layer_Number, self.Frozen_Selector, sample_id, self.Class_Name)

        self.confusion_matrices = self._get_confusion_matrices(self.Labelset_Name,
                                                               self.Model_Selector,
                                                               self.Frozen_Selector)

        if cache_result is None:
            with requests.Session() as s:
                result = s.get(self.server_address + 'cam/{0:}/{1:}/{2:}/{3:}/{4:}/{5:}'.format(self.Labelset_Name,
                                                                                                  self.Model_Selector,
                                                                                                  self.Network_Layer_Number,
                                                                                                  self.Frozen_Selector,
                                                                                                  sample_id,
                                                                                                  self.Class_Name))
                r = loads(result.text)
                map = np.array(r['map'])
                probs = r['y_prob']

                self.cache.append({'labelset_Name': self.Labelset_Name,
                                   'Model_Selector': self.Model_Selector,
                                   'Network_Layer_Number': self.Network_Layer_Number,
                                   'Frozen_Selector': self.Frozen_Selector,
                                   'sample_id': sample_id,
                                   'Class_Name': self.Class_Name,
                                   'probs': probs,
                                   'map': map})
        else:
            probs = cache_result['probs']
            map = cache_result['map']

        self.probs = probs
        return map, probs

    @param.depends('Image_Name', 'Show_CAM', 'Alpha')
    def load_symbol(self):
        logger.info("Get load_symbol")
        """
        Creates the CAM and Thin section holoviews objects to plot in the viewer.
        """
        X_np = self.load_image()
        resize = transforms.Resize((X_np.shape[0], X_np.shape[1]))
        map, _ = self.make_map()
        map = resize(torch.from_numpy(map).unsqueeze(0)).numpy()[0]
        hv_thinsection = create_holoviews_thinsection(X_np[::-1])

        hv_cam = create_holoviews_cam(map.T).opts(alpha=self.Alpha, cmap='inferno')

        image = hv_thinsection

        if self.Show_CAM:
            image *= hv_cam

        return image

    def bar_plot(self):
        logger.info("Get bar_plot")
        """
        Creates the bar plot for image probabilities
        """
        data = [(clas, prob) for clas, prob in zip(self.param['Class_Name'].objects, self.probs)]
        bars = hv.Bars(data, hv.Dimension('Class Name'), 'Probability')

        return bars

    #@pn.depends('Class_Name')
    def plot_confusion_matrix_train(self):
        class_names = self.param['Class_Name'].objects

        fig, axarr = plt.subplots(2, 1, figsize=(6, 14))
        for ax, phase in zip(axarr, ["train", "test"]):
            df_cm = pd.DataFrame(self.confusion_matrices[phase], class_names, class_names)
            sn.set(font_scale=1.4)  # for label size
            sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, ax=ax)  # font size
            ax.set_ylabel("Ground Truth Label", fontsize=16)
            ax.set_xlabel("Predicted Label", fontsize=16)
            ax.set_title(phase, fontsize=18)
        fig.savefig("tmp.png", bbox_inches="tight")
        return pn.panel('tmp.png', width=300)

    def view(self):
        logger.info("Get view")
        """
        Initializes the viewer and provides the dynamically updating holoviews image viewer.
        """
        self.make_map()
        thin_section = hv.DynamicMap(self.map)

        thin_section = rasterize(thin_section).opts(data_aspect=1.0, frame_height=600, frame_width=1000,
                                                    active_tools=['xwheel_zoom', 'pan'])

        col = pn.Row(thin_section, pn.Column(self.bar_plot(), self.plot_confusion_matrix_train()))
        return col
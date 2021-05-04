import param
from holoviews.operation.datashader import rasterize
import numpy as np
from neural_rock.cam import GradCam, make_maps
from neural_rock.app.utils import load_model, load_data, get_train_test_split, Model, compute_images, MEAN_TRAIN, STD_TRAIN
from neural_rock.app.plot import create_holoviews_thinsection, create_holoviews_cam
from neural_rock.app.viewer import ThinSectionViewer
from imageio import imread
import panel as pn
import holoviews as hv
from neural_rock.app.viewer import ThinSectionViewer
from awesome_panel_extensions.frameworks.fast import FastTemplate
hv.extension('bokeh')

"""
labels_sets = ['Dunham', 'DominantPore', 'Lucia']

train_test_split = get_train_test_split()

labelset = 'Dunham'
device = 'cpu'

image_paths, modified_label_map, image_names, class_names = load_data(labelset)

for id, img in image_paths.items():
    if id in train_test_split['train']:
        img['train'] = 'Train'
    else:
        img['train'] = 'Test'

if labelset == "Lucia":
    chkpt = "./data/models/Lucia/v1/epoch=29-step=629.ckpt"
elif labelset == "DominantPore":
    chkpt = "./data/models/DominantPore/v1/epoch=0-step=20.ckpt"
elif labelset == "Dunham":
    chkpt = "./data/models/Dunham/v1/epoch=11-step=251.ckpt"
model = load_model(chkpt, num_classes=len(class_names))

image_name_map = {}
image_names = []
for image_name, dset in image_paths.items():
    image_n = "-".join([str(image_name), dset['label'], str(dset['train'])])
    image_names.append(image_n)
    image_name_map[image_n] = image_name

image_id = image_names[0]

image_id = image_name_map[image_id]

min_layer = 0
max_layer = len(list(model.feature_extractor.modules()))-2

class_n = class_names[0]
class_idx = np.argwhere(np.array(class_names) == class_n)[0, 0]

model = Model(model.feature_extractor, model.classifier)
model.to(device)
grad_cam = GradCam(model=model.to(device),
                   feature_module=model.feature_extractor.to(device),
                   target_layer_names=[str(max_layer)],
                   device=device)
"""

label_set_name = param.ObjectSelector(default="Lucia", objects=["Dunham", 'Lucia', 'DominantPore'])
device = param.ObjectSelector(default="cpu", objects=["cpu", 'cuda'])
model_selector = param.ObjectSelector(default="VGG", objects=['VGG', 'ResNet'])
frozen_selector = param.ObjectSelector(default="True", objects=['True', 'False'])

Network_Layer_Number = param.Integer(default=0, bounds=(0, 1))

Class_Name = param.ObjectSelector(default="default", objects=["default"])

Image_Name = param.ObjectSelector(default="default", objects=["default"])

theme = pn.template.react.DarkTheme
explorer = ThinSectionViewer()
layout_explorer = pn.Row(pn.Column(
    explorer.param.label_set_name,
    explorer.param.device,
    explorer.param.model_selector,
    explorer.param.frozen_selector,
    explorer.param.Class_Name,
    explorer.param.Image_Name,
    explorer.param.Network_Layer_Number),
    pn.Column(explorer.view, sizing_mode='stretch_width'),
    sizing_mode='stretch_width')

css = '''
.bk.bk-input {
    background-color: dimgray;
}'''

pn.extension(raw_css=[css])

layout = pn.template.ReactTemplate(
    title="Neural Rock Viewer",
    theme=theme,
    row_height=200,
)
layout.main[:, :] = layout_explorer

layout.servable()

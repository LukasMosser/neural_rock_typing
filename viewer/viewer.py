import panel as pn
import holoviews as hv
from pathlib import Path
import os
from neural_rock.app.viewer import ThinSectionViewer
from neural_rock.server.utils import make_label_sets, init_model_zoo, model_lookup, load_image
from neural_rock.data_models import make_image_dataset, ModelName, LabelSetName, LabelSets, CAMRequest
from neural_rock.preprocess import load_label_dataframe

hv.extension('bokeh')

device = 'cpu'

valid_layers = {'resnet': list(range(8)),
                'vgg': list(range(21))}

base_path = Path(os.getenv('WORKDIR'))
os.chdir(os.getenv('WORKDIR'))

server_address = "http://{0:}:8000/".format(os.getenv('APIHOST'))

df = load_label_dataframe(base_path=base_path)
image_dataset = make_image_dataset(df, base_path=base_path)
label_sets = make_label_sets(df)
model_zoo = init_model_zoo(base_path=base_path)

theme = pn.template.react.DarkTheme
explorer = ThinSectionViewer(server_address, image_dataset, label_sets, model_zoo)
layout_explorer = pn.Row(pn.Column(
    explorer.param.label_set_name,
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

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

theme = pn.template.react.DarkTheme
explorer = ThinSectionViewer(server_address, image_dataset)

control_app = pn.Param(
    explorer.param,
    parameters=["Labelset_Name",
                "Model_Selector",
                "Frozen_Selector",
                "Network_Layer_Number",
                "Class_Name",
                "Image_Name"],
    show_name=True
)

layout_explorer = pn.Row(
    pn.Column(control_app),
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

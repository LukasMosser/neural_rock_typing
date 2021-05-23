import panel as pn
import holoviews as hv
from pathlib import Path
import os
from neural_rock.app.viewer import ThinSectionViewer
from neural_rock.data_models import make_image_dataset
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
                "Image_Name",
                "Show_CAM",
                "Alpha"],
    show_name=True
)

howto_str = """
### How-To:
Heatmap shows neural network activation 
response with respect to the selected class label. 

Select different network types to see impact 
of different architecture and learning processes.

See [here](https://distill.pub/2020/attribution-baselines/) for a great article on 
feature visualization in CNNs.

#### Model Selector
Choose different CNN Architecture: Resnet or VGG

#### Network Layer Number
Select which layer of network to show activations for.

#### Frozen Selector
CNN Feature Extractor Active (False), In-active (True)

### Authors:
[Lukas Mosser](https://www.linkedin.com/in/lukas-mosser), [George Ghon](https://www.linkedin.com/in/george-g-30015639/), [Gregor Baechle](https://www.linkedin.com/in/gbaechle/)
"""
layout_explorer = pn.Column(
    pn.Row(pn.Column(control_app), explorer.view),
    pn.Row(pn.pane.Markdown(howto_str, width=800)), sizing_mode='stretch_width')
css = '''
.bk.bk-input {
    background-color: dimgray;
}'''

pn.extension(raw_css=[css])

layout = pn.template.ReactTemplate(
    title="Neural Rock Viewer",
    theme=theme,
)
layout.main[:, :] = layout_explorer

layout.servable()

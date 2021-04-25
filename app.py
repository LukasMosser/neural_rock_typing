import streamlit as st
import torch.nn as nn
import numpy as np
from torchvision import transforms
import torch
from torch.autograd import Variable
from neural_rock.dataset import ThinSectionDataset
from neural_rock.cam import GradCam
import matplotlib.pyplot as plt
import altair as alt
import pandas as pd
import albumentations as A
import imageio

import os
import argparse
import streamlit as st
import json
import xarray as xa
import holoviews as hv
from holoviews.operation.datashader import rasterize
hv.extension('bokeh')

with open("./data/train_test_split.json") as f:
  train_test_split = json.load(f)

parser = argparse.ArgumentParser(description='This app lists animals')

parser.add_argument('--worker', type=str, default="cpu", choices=["cpu", "cuda"],
                    help="What type of worker")
try:
    args = parser.parse_args()
except SystemExit as e:
    # This exception will be raised if --help or invalid command line arguments
    # are used. Currently streamlit prevents the program from exiting normally
    # so we have to do a hard exit.
    os._exit(e.code)

st.set_page_config(layout='wide', page_title='Thin Section Neural Visualizer')

st.title('Thin Section Neural Visualizer')

MEAN_TRAIN = np.array([0.485, 0.456, 0.406])
STD_TRAIN = np.array([0.229, 0.224, 0.225])

def load_img(path):
    image = imageio.imread(path).astype(np.float32)
    return image


@st.cache(allow_output_mutation=True)
def load_data(problem):
    transform = A.Compose([A.RandomCrop(width=512, height=512),
                           A.Resize(width=224, height=224),
                           A.Normalize()])

    train_dataset = ThinSectionDataset("./data/Images_PhD_Miami/Leg194", problem,
                                       transform=transform, train=True, seed=42, preload_images=False)
    val_dataset = ThinSectionDataset("./data/Images_PhD_Miami/Leg194", problem,
                                     transform=transform, train=False, seed=42, preload_images=False)

    idx_check = [train_id in val_dataset.image_ids for train_id in train_dataset.image_ids]
    assert not any(idx_check)

    modified_label_map = train_dataset.modified_label_map
    image_names = train_dataset.image_ids
    class_names = train_dataset.class_names

    paths = {}
    for train, dataset in zip([True, False], [train_dataset.dataset, val_dataset.dataset]):
        for key, dset in dataset.items():
            paths[key] = {'path': dset['path'],
                          'label': dset[problem],
                          'mask_path': dset['mask_path'], 'train': "Train" if train else "Test"}

    return paths, modified_label_map, image_names, class_names


@st.cache(allow_output_mutation=True)
def load_model(checkpoint, num_classes=5):
    from neural_rock.model import NeuralRockModel
    model = NeuralRockModel.load_from_checkpoint(checkpoint, num_classes=num_classes)
    model.eval()
    model.freeze()
    return model


def compute_images(X, grad_cam, max_classes, resize, device="cpu"):
    maps = []
    for i in range(max_classes):
        gb = grad_cam(X.to(device), i)
        gb = (gb - np.min(gb)) / (np.max(gb) - np.min(gb))
        gb = resize(torch.from_numpy(gb).unsqueeze(0))[0]
        maps.append(gb.cpu().numpy())
    maps = np.stack(maps, axis=0)
    return maps


device = args.worker

data_load_state = st.text('Loading data...')

data_load_state.text("Done! (using st.cache)")

col1, col2 = st.beta_columns((1, 1))

with col1:
    problem = st.selectbox("Choose Classifier", ['Dunham', 'DominantPore', 'Lucia'])

    image_paths, modified_label_map, image_names, class_names = load_data(problem)

    for id, img in image_paths.items():
        if id in train_test_split['train']:
            img['train'] = 'Train'
        else:
            img['train'] = 'Test'

    if problem == "Lucia":
        chkpt = "./data/models/Lucia/v1/epoch=29-step=629.ckpt"
    elif problem == "DominantPore":
        chkpt = "./data/models/DominantPore/v1/epoch=0-step=20.ckpt"
    elif problem == "Dunham":
        chkpt = "./data/models/Dunham/v1/epoch=11-step=251.ckpt"
    model = load_model(chkpt, num_classes=len(class_names))

    image_name_map = {}
    image_names = []
    for image_name, dset in image_paths.items():
        image_n = "-".join([str(image_name), dset['label'], str(dset['train'])])
        image_names.append(image_n)
        image_name_map[image_n] = image_name

    image_id = st.selectbox("Choose an Image", image_names)

    image_id = image_name_map[image_id]

    layer = st.slider("Which Layer to Visualize", min_value=0, max_value=len(list(model.feature_extractor.modules()))-2, value=len(list(model.feature_extractor.modules()))-2, step=1)
    class_n = st.selectbox("Which Class to Visualize", class_names)
    class_idx = np.argwhere(np.array(class_names) == class_n)[0, 0]

    class Model(nn.Module):
        def __init__(self, feature_extractor, fc):
            super(Model, self).__init__()
            self.feature_extractor = feature_extractor
            self.fc = fc

        def forward(self, x):
            x = self.feature_extractor(x)
            x = self.fc(x)
            return x

    model = Model(model.feature_extractor, model.classifier)
    model.to(device)
    grad_cam = GradCam(model=model.to(device),
                       feature_module=model.feature_extractor.to(device),
                       target_layer_names=[str(layer)],
                       device=device)

    from imageio import imread
    X = imread(image_paths[image_id]['path'])
    mask = imread(image_paths[image_id]['mask_path'])
    mask = (1 - (mask > 0).astype(np.uint8))
    X = X * np.repeat(np.expand_dims(mask, axis=-1), 3, axis=-1)

    label = image_paths[image_id]['label']

    transform = A.Compose([
        A.CenterCrop(1024, 1024),
        A.Resize(224*2, 224*2),
        A.Normalize()])

    resize = transforms.Resize((224*2, 224*2))

    X = transform(image=X)['image'].transpose(2, 0, 1)

    X = Variable(torch.from_numpy(X).unsqueeze(0), requires_grad=True).to(device)

    with torch.no_grad():
        output = model(X).to(device)

    image_patch = np.transpose(X.data.cpu().numpy()[0], (1, 2, 0)) * STD_TRAIN + MEAN_TRAIN
    X = X.to(device)
    maps = compute_images(X, grad_cam, len(class_names), resize=resize, device=device)

img_np = imageio.imread("./data/Leg194/1x/Leg194_2-1x.jpg")
print(img_np.shape)
coords = {'x': np.arange(img_np.shape[0]), 'y': np.arange(img_np.shape[1])}
print(len(coords['x']), len(coords['y']))
#image_patch = image_patch.reshape(image_patch.shape[0], image_patch.shape[1], 1, 1, 1)
#xr_ds = xa.DataArray(name='rgb', data=image_patch, coords=coords, dims=["x", "y"])

r_ = xa.DataArray(name='r', data=img_np[..., 0], coords=coords, dims=['x', 'y'])
g_ = xa.DataArray(name='g', data=img_np[..., 1], coords=coords, dims=['x', 'y'])
b_ = xa.DataArray(name='b', data=img_np[..., 2], coords=coords, dims=['x', 'y'])

# Convert to stack of images with x/y-coordinates along axes
#image_stack = ds.to(hv.RGB, kdims=['x', 'y'], vdims=['r', 'g', 'b'])
from datashader.utils import ngjit
import datashader as ds
nodata= 1

@ngjit
def normalize_data(agg):
    out = np.zeros_like(agg)
    min_val = 0
    max_val = 2**16 - 1
    range_val = max_val - min_val
    col, rows = agg.shape
    c = 40
    th = .125
    for x in range(col):
        for y in range(rows):
            val = agg[x, y]
            norm = (val - min_val) / range_val
            norm = 1 / (1 + np.exp(c * (th - norm))) # bonus
            out[x, y] = norm * 255.0
    return out

def combine_bands(r, g, b):
    xs, ys = r['y'], r['x']
    r, g, b = [ds.utils.orient_array(im) for im in (r, g, b)]
    print(xs.shape, ys.shape, r.shape, g.shape, b.shape)
    #a = (np.where(np.logical_or(np.isnan(r),r<=nodata),0,255)).astype(np.uint8)
    r = r#(normalize_data(r)).astype(np.uint8)
    g = g#(normalize_data(g)).astype(np.uint8)
    b = b#(normalize_data(b)).astype(np.uint8)
    print(xs.shape, ys.shape, r.shape, g.shape, b.shape)
    return hv.RGB((xs, ys[::-1], r, g, b), vdims=list('RGB'))

rgb = combine_bands(r_, g_, b_)
# Apply regridding if each image is large
regridded = rasterize(rgb)

# Set a global Intensity range
#regridded = regridded.redim.range(Intensity=(0, 100))
regridded = regridded.opts(responsive=False, width=800, height=800, active_tools=['xwheel_zoom', 'pan']) #, width=800, height=800,

with col1:
    st.bokeh_chart(hv.render(regridded, backend='bokeh'))

with col1:

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].set_title("Image Class {0:}".format(label))
    ax[1].set_title("CAM Activation {0:}".format(class_names[class_idx]))
    ax[0].imshow(image_patch.transpose((1, 0, 2)))
    ax[1].imshow(image_patch.transpose((1, 0, 2)))
    ax[1].imshow(maps[class_idx].T, cmap="inferno", alpha=0.5)
    for a in ax:
        a.set_axis_off()
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots(len(class_names), 3, figsize=(3 * 6, 6 * len(class_names)))

    for i in range(len(class_names)):
        gb = maps[i]

        ax[i, 0].set_title("Image Class {0:}".format(label))
        ax[i, 1].set_title("CAM Activation {0:}".format(class_names[i]))
        ax[i, 2].set_title("CAM Activation Overlay {0:}".format(class_names[i]))
        ax[i, 0].imshow(image_patch.transpose((1, 0, 2)))
        ax[i, 1].imshow(gb.T, cmap="inferno")

        ax[i, 2].imshow(image_patch.transpose((1, 0, 2)))
        ax[i, 2].imshow(gb.T, cmap="inferno", alpha=0.5)
        for a in ax[i]:
            a.set_axis_off()
    st.pyplot(fig)

    source = pd.DataFrame({
        'class': class_names,
        'probability': torch.softmax(output, dim=1).cpu().numpy().flatten()
    })

    c = alt.Chart(source).mark_bar().encode(
        x='class',
        y='probability'
    )

with col1:
    st.altair_chart(c, use_container_width=True)







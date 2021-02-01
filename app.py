import streamlit as st
import pandas as pd
import numpy as np
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch
from torch.autograd import Variable
from neural_rock.utils import load_checkpoint
from neural_rock.preprocess import load_excel, load_images_w_rows_in_table, make_feature_map, make_feature_map_dunham
from neural_rock.dataset import ThinSectionDataset
from neural_rock.cam import GradCam
import matplotlib.pyplot as plt
import altair as alt
import pandas as pd

st.title('Thin Section Neural Visualizer')

MEAN_TRAIN = [0.38162467, 0.4421087, 0.40902838]
STD_TRAIN = [0.21275578, 0.16988535, 0.21243732]


@st.cache(allow_output_mutation=True)
def load_data():
    df = load_excel()

    imgs, features, df_ = load_images_w_rows_in_table(df)
    image_names = [feat[0] for feat in features]
    pore_type, modified_label_map, class_names = make_feature_map_dunham(features)

    labels = torch.from_numpy(np.array(pore_type))

    imgs = torch.from_numpy(np.transpose(imgs, (0, 3, 1, 2)))
    resize = transforms.Resize((imgs.shape[2]//2, imgs.shape[3]//2))
    transform = transforms.Compose([transforms.Normalize(MEAN_TRAIN, STD_TRAIN),
                                   resize
                                    ])

    dataset = ThinSectionDataset(imgs, labels, class_names, transform=transform)
    return dataset, modified_label_map, resize, image_names, class_names


@st.cache(allow_output_mutation=True)
def load_model():
    model = models.alexnet(pretrained=True)

    model.classifier = nn.Sequential(
            nn.Dropout(p=0.9),
            nn.Linear(9216, 128),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, 64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, 4))

    model, _, _ = load_checkpoint("./runs/_2021-01-28_20-28-29/checkpoints/best.pth", model)
    model.eval()
    return model


@st.cache(allow_output_mutation=True)
def compute_images(X, grad_cam, max_classes):
    maps = []
    for i in range(max_classes):
        gb = grad_cam(X, i)
        gb = (gb - np.min(gb)) / (np.max(gb) - np.min(gb))
        gb = resize(torch.from_numpy(gb).unsqueeze(0))[0]
        maps.append(gb)
    return maps


model = load_model()
data_load_state = st.text('Loading data...')
dataset, modified_label_map, resize, image_names, class_names = load_data()

data_load_state.text("Done! (using st.cache)")

image_id = st.sidebar.selectbox("Choose an Image", image_names)
layer = st.sidebar.slider("Which Layer to Visualize", min_value=0, max_value=13, value=11, step=1)
class_n = st.sidebar.selectbox("Which Class to Visualize", class_names)
class_idx = np.argwhere(np.array(class_names) == class_n)[0, 0]

index = np.argwhere(np.array(image_names) == image_id)[0, 0]

grad_cam = GradCam(model=model,
                   feature_module=model.features,
                   target_layer_names=[str(layer)],
                   use_cuda=False)

X, lab = dataset[index]
X = Variable(X.unsqueeze(0), requires_grad=True)

with torch.no_grad():
    output = model(X)

image_patch = np.transpose(X.data.numpy()[0], (1, 2, 0)) * STD_TRAIN + MEAN_TRAIN
maps = compute_images(X, grad_cam, len(class_names))

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].set_title("Image Class {0:}".format(class_names[lab]))
ax[1].set_title("CAM Activation {0:}".format(class_names[class_idx]))
ax[0].imshow(image_patch)
ax[1].imshow(image_patch)
ax[1].imshow(maps[class_idx], cmap="inferno", alpha=0.5)
for a in ax:
    a.set_axis_off()
st.pyplot(fig)

fig, ax = plt.subplots(len(class_names), 3, figsize=(3 * 6, 6 * len(class_names)))

for i in range(len(class_names)):
    gb = maps[i]

    ax[i, 0].set_title("Image Class {0:}".format(class_names[lab]))
    ax[i, 1].set_title("CAM Activation {0:}".format(class_names[i]))
    ax[i, 2].set_title("CAM Activation Overlay {0:}".format(class_names[i]))
    ax[i, 0].imshow(image_patch)
    ax[i, 1].imshow(gb, cmap="inferno")

    ax[i, 2].imshow(image_patch)
    ax[i, 2].imshow(gb, cmap="inferno", alpha=0.5)
    for a in ax[i]:
        a.set_axis_off()
st.pyplot(fig)

source = pd.DataFrame({
    'a': class_names,
    'b': torch.softmax(output, dim=1).numpy().flatten()
})

c = alt.Chart(source).mark_bar().encode(
    x='a',
    y='b'
)

st.altair_chart(c, use_container_width=True)





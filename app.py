import streamlit as st
import pandas as pd
import numpy as np
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch
from torch.autograd import Variable
from neural_rock.utils import load_checkpoint
from neural_rock.preprocess import load_excel, load_images_w_rows_in_table, load_image_names_in_rows, make_feature_map_microp, make_feature_map_dunham, make_feature_map_lucia
from neural_rock.dataset import ThinSectionDataset
from neural_rock.cam import GradCam
import matplotlib.pyplot as plt
import altair as alt
import pandas as pd
import albumentations as A
from sklearn.model_selection import StratifiedShuffleSplit
import imageio

st.set_page_config(layout='wide', page_title='Thin Section Neural Visualizer')

st.title('Thin Section Neural Visualizer')

MEAN_TRAIN = np.array([0.485, 0.456, 0.406])
STD_TRAIN = np.array([0.229, 0.224, 0.225])


def load_img(path):
    image = imageio.imread(path).astype(np.float32)
    return image

@st.cache(allow_output_mutation=True)
def load_data(type):
    df = load_excel()

    imgs, features, df_ = load_image_names_in_rows(df)

    if type == 'Dunham':
        pore_type, modified_label_map, class_names = make_feature_map_dunham(features)
        checkpoint = "./runs/Dunham_2021-03-07_17-55-41/checkpoints/best.pth"

    elif type == 'Micro Porosity':
        pore_type, modified_label_map, class_names = make_feature_map_microp(features)
        checkpoint = "./runs/DominantPore_2021-03-10_17-58-37/checkpoints/best.pth"

    else:
        pore_type, modified_label_map, class_names = make_feature_map_lucia(features)
        checkpoint = "./runs/Lucia_2021-03-10_19-12-49/checkpoints/best.pth"

    splitter = StratifiedShuffleSplit(test_size=0.50, random_state=42)

    for train_idx, val_idx in splitter.split(imgs, pore_type):
        print(train_idx, val_idx)
        break

    image_names = []
    for i, feat in enumerate(features):
        if i in train_idx:
            image_names.append('{} - train - {}'.format(feat[0], class_names[pore_type[i]]))
        else:
            image_names.append('{} - validation - {}'.format(feat[0], class_names[pore_type[i]]))

    labels = torch.from_numpy(np.array(pore_type))

    return imgs,  labels, modified_label_map, image_names, class_names, checkpoint


@st.cache(allow_output_mutation=True)
def load_model(num_classes, checkpoint):
    model = models.vgg11(pretrained=True)

    model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(25088, 1024, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 256, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, num_classes, bias=True))
    print(checkpoint)
    model, _, _ = load_checkpoint(checkpoint, model)
    model.eval()
    return model


def compute_images(X, grad_cam, max_classes, resize):
    maps = []
    for i in range(max_classes):
        gb = grad_cam(X, i)
        gb = (gb - np.min(gb)) / (np.max(gb) - np.min(gb))
        gb = resize(torch.from_numpy(gb).unsqueeze(0))[0]
        maps.append(gb.cpu().numpy())
    maps = np.stack(maps, axis=0)
    return maps


device = "cuda"

data_load_state = st.text('Loading data...')

data_load_state.text("Done! (using st.cache)")

col1, col2 = st.beta_columns((1, 1))

with col1:
    problem = st.selectbox("Choose Classifier", ['Dunham', 'Micro Porosity', 'Lucia'])

    image_paths, labels, modified_label_map, image_names, class_names, checkpoint = load_data(problem)

    image_id = st.selectbox("Choose an Image", image_names)
    print(checkpoint)
    model = load_model(len(class_names), checkpoint).to(device)
    layer = st.slider("Which Layer to Visualize", min_value=0, max_value=len(list(model.features.modules()))-2, value=len(list(model.features.modules()))-2, step=1)
    class_n = st.selectbox("Which Class to Visualize", class_names)
    class_idx = np.argwhere(np.array(class_names) == class_n)[0, 0]

    index = np.argwhere(np.array(image_names) == image_id)[0, 0]

    grad_cam = GradCam(model=model,
                       feature_module=model.features,
                       target_layer_names=[str(layer)],
                       use_cuda=True if device is "cuda" else False)

    X = load_img(image_paths[index])
    lab = labels[index]

    transform = A.Compose([A.Normalize(mean=MEAN_TRAIN, std=STD_TRAIN),
                           A.Resize(X.shape[0] // 2, X.shape[1] // 2)])

    resize = transforms.Resize((X.shape[0]//2,  X.shape[1]//2))

    X = transform(image=X)['image'].transpose(2, 0, 1)


    X = Variable(torch.from_numpy(X).unsqueeze(0), requires_grad=True).to(device)

    with torch.no_grad():
        output = model(X).to(device)

    image_patch = np.transpose(X.data.cpu().numpy()[0], (1, 2, 0)) * STD_TRAIN + MEAN_TRAIN

    maps = compute_images(X, grad_cam, len(class_names), resize=resize)

with col1:

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].set_title("Image Class {0:}".format(class_names[lab]))
    ax[1].set_title("CAM Activation {0:}".format(class_names[class_idx]))
    ax[0].imshow(image_patch)
    ax[1].imshow(image_patch)
    ax[1].imshow(maps[class_idx], cmap="inferno", alpha=0.5)
    for a in ax:
        a.set_axis_off()
    st.pyplot(fig)

with col2:
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
        'class': class_names,
        'probability': torch.softmax(output, dim=1).cpu().numpy().flatten()
    })

    c = alt.Chart(source).mark_bar().encode(
        x='class',
        y='probability'
    )

with col1:
    st.altair_chart(c, use_container_width=True)





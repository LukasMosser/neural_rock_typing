import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import imageio


def load_excel():
    columns = ['Location', 'Sample']
    features = ['Mineralogy', 'Dunham_class', 'Lucia_class', 'Macro_Dominant_type', 'Macro_Minor_types']
    df_dry = pd.read_excel("./data/Data_Sheet_GDrive.xls", "Chapter_3_dry", skiprows=[1])[columns+features]
    df_dry['saturation'] = 'dry'
    df_wet = pd.read_excel("./data/Data_Sheet_GDrive.xls", "Chapter_3_water saturated", skiprows=[1])[columns+features]
    df_wet['saturation'] = 'wet'
    df = pd.concat([df_dry, df_wet])
    df = df.loc[df["Location"]=='Leg194']
    df['Sample'] = df['Sample'].astype(int)
    return df


def split_fname(fname):
    parts = fname.split('_')
    idx_mag_file = parts[1]
    idx_mag_file_split = idx_mag_file.split("-")
    idx = int(idx_mag_file_split[0])
    try:
        mag = idx_mag_file_split[1].split(".")[0]
    except IndexError:
        print(idx_mag_file_split)
    return idx, mag


def load_images_w_rows_in_table(df, label_idx=None, filter_labels=[]):
    image_idxs = {}
    direc = "./data/Leg194/1x"
    for fname in os.listdir(direc):
        if fname not in ['WS_FTP.LOG', 'Thumbs.db']:
            idx, mag = split_fname(fname)
            if mag in ['1x', '1X'] and idx in df['Sample'].unique():
                image_idxs[idx] = fname

    df_ = df[df['Sample'].isin(image_idxs.keys())]

    imgs = []
    features = []
    for idx, row in tqdm(df_.iterrows()):
        img = imageio.imread(os.path.join(direc, image_idxs[row[1]]))
        if label_idx is not None:
            if row[label_idx] not in filter_labels:
                imgs.append(img)
                features.append(row[1:-1])
        else:
            imgs.append(img)
            features.append(row[1:-1])

    imgs = np.array(imgs).astype(np.float32)/255.
    return imgs, features, df_


def make_feature_map(features):
    pore_type = [val[4] for val in features]
    modified_label_map = {'IP': 0, 'IX': 1, 'MO': 2, 'VUG': 3, 'WP': 4}
    class_names = list(modified_label_map.keys())
    return pore_type, modified_label_map, class_names


def make_feature_map_dunham(features):
    modified_label_map = {' rDol': 0, 'B': 1, 'B-G': 1, 'FL': 1,
                          'G': 2, 'G-B': 2, 'G-P': 2, 'P': 3, 'P-G': 3, 'rDol': 0}
    pore_type = [modified_label_map[val[2]] for val in features]
    print(np.unique(pore_type, return_counts=True))

    class_names = ['rDol', 'B', 'G', 'P']
    return pore_type, modified_label_map, class_names


def create_images_and_labels(imgs, pore_type):
    images_np = []
    labels_np = []
    for label, img in zip(pore_type, imgs):
        img = np.transpose(img, (2, 0, 1))
        shape = img.shape

        img1 = img[:, :shape[1]//2, :shape[2]//2]
        img2 = img[:, shape[1]//2:, :shape[2]//2]
        img3 = img[:, :shape[1]//2, shape[2]//2:]
        img4 = img[:, shape[1]//2:, shape[2]//2:]

        for im in [img1, img2, img3, img4]:
            images_np.append(im)
            labels_np.append(label)

    images_np = np.array(images_np)
    labels_np = np.array(labels_np)
    return images_np, labels_np
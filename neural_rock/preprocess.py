import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import imageio


def load_excel():
    columns = ['Location', 'Sample']
    features = ['Mineralogy', 'Dunham_class', 'Lucia_class', 'Macro_Dominant_type', 'Macro_Minor_types']
    df_dry = pd.read_excel("./data/Data_Sheet_GDrive_updated.xls", "Chapter_3_dry", skiprows=[1])[columns+features]
    df_dry['saturation'] = 'dry'
    df_wet = pd.read_excel("./data/Data_Sheet_GDrive_updated.xls", "Chapter_3_water_saturated", skiprows=[1])[columns+features]
    df_wet['saturation'] = 'wet'
    df = pd.concat([df_dry, df_wet])
    df = df.loc[df["Location"] == 'Leg194']
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


def load_images(path, img_type="Xppl", size="1x"):
    imgs = {}
    for fname in os.listdir(os.path.join(path, "img")):
        name_num, type, ending = fname.split(".")
        mask_path = os.path.join(path, "ROI", "ROI_"+name_num+".png")#Leg194_72-1x.png
        if not os.path.exists(mask_path):
            continue

        num, mag = name_num.split("_")[1].split("-")
        if type.lower() == img_type.lower() and mag.lower() == size.lower():
            imgs[int(num)] = {"path": os.path.join(path, "img", fname),
                              "mask_path": mask_path,
                              "mag": mag.lower(),
                              "type": type.lower()}


    return imgs


def load_features(images, df):
    df_ = df[df['Sample'].isin(images.keys())]

    for idx, row in df_.iterrows():
        num = int(row["Sample"])
        images[num]["features"] = row[1:-1]
        images[num]["Lucia"] = images[num]["features"][3]
        images[num]["Dunham"] = images[num]["features"][2]
        images[num]["DominantPore"] = images[num]["features"][4]
    return images, df_


def load_image_names_in_rows(df):
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
    for idx, row in df_.iterrows():
        imgs.append(os.path.join(direc, image_idxs[row[1]]))
        features.append(row[1:-1])

    return imgs, features, df_


def make_feature_map_microp(features):
    modified_label_map = {'IP': 0, 'VUG': 1, 'MO': 2, 'IX': 3, 'WF': 4, 'WP': 4}
    pore_type = [modified_label_map[val[4]] for val in features]
    class_names = ['IP', 'VUG', 'MO', 'IX', 'WF-WP']
    return pore_type, modified_label_map, class_names


def make_feature_map_lucia(features):
    modified_label_map = {'0': 0, '1': 1, '2': 2}
    pore_type = [modified_label_map[str(val[3])] for val in features]
    class_names = ['0', '1', '2']
    return pore_type, modified_label_map, class_names


def make_feature_map_dunham(features):
    print(np.unique([val[2] for val in features]))
    modified_label_map = {'rDol': 0, 'B': 1, 'FL': 2,
                          'G': 3, 'G-P': 4, 'P': 5, 'P-G': 4}
    pore_type = [modified_label_map[val[2]] for val in features]
    print(np.unique(pore_type, return_counts=True))

    class_names = ['rDol', 'B', 'FL', 'G', 'G-P,P-G', 'P']
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
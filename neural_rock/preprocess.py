import pandas as pd
import os
from pathlib import Path
from typing import List


def get_leg194_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pandas pipe function to reduce only to Leg194 samples
    """
    df = df.loc[df['Location'] == 'Leg194']
    df = df.loc[df['Sample'].apply(lambda x: str(x).isdigit())]
    df['Sample'] = df['Sample'].astype(int)
    return df


def get_class_label_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pandas pipe function to get only relevant data columns from data frame
    """
    columns = ['Location', 'Sample', 'Dunham_class', 'Lucia_class', 'Macro_Dominant_type']
    return df[columns]


def merge(df1: pd.DataFrame,
          df2: pd.DataFrame) -> pd.DataFrame:
    """
    Pandas pipe function to merge two dataframes
    """
    df = df1.merge(df2, on=list(df1.columns), how='left')
    return df


def drop_no_image(df: pd.DataFrame,
                  imaged_samples: List[int]) -> pd.DataFrame:
    """
    Pandas pipe function to drop any rows from table for samples that have no images.
    """
    df_temp = df[df['Sample'].isin(imaged_samples)]
    return df_temp


def get_image_paths(base_path: Path = Path(".."),
                    rel_path: Path = Path("data/Images_PhD_Miami/Leg194/"),
                    imaging: str = "Xppl"):
    """
    Gets all the local images paths for the ROI and the Imaged Thin Sections
    """
    roi_ids = set([int(fname.split("_")[2].split("-")[0]) for fname in os.listdir(base_path.joinpath(rel_path, "ROI"))])
    img_ids = []
    img_paths = {}
    for fname in os.listdir(base_path.joinpath(rel_path, "img")):
        if fname.split(".")[1] == imaging:
            sample_id = int(fname.split("-")[0].split("_")[1])
            img_ids.append(sample_id)
            img_paths[sample_id] = base_path.joinpath(rel_path, "img", fname)
    img_ids = set(img_ids)

    sample_ids = roi_ids.intersection(img_ids)

    img_paths = {k: path for k, path in img_paths.items() if k in sample_ids}

    roi_paths = {}
    for fname in os.listdir(base_path.joinpath(rel_path, "ROI")):
        sample_id = int(fname.split("_")[2].split("-")[0])
        if sample_id in sample_ids:
            roi_paths[sample_id] = base_path.joinpath(rel_path, "ROI", fname)
    return sample_ids, img_paths, roi_paths


def load_label_dataframe(base_path: Path=Path("..")) -> pd.DataFrame:
    """
    Data Preprocessing function to load the Leg194 dataset.
    Uses pandas pipes to filter dataframe based on available images.
    """
    sample_ids, image_paths, roi_paths = get_image_paths(base_path=base_path)
    excel_path = base_path.joinpath("data/Data_Sheet_GDrive_new.xls")
    df_dry = pd.read_excel(excel_path, sheet_name="Chaper_3_dry")
    df_dry = df_dry.pipe(get_leg194_data).pipe(get_class_label_columns)

    df_wet = pd.read_excel(excel_path, sheet_name="Chapter_3_water saturated")
    df_wet = df_wet.pipe(get_leg194_data).pipe(get_class_label_columns)

    df_label = df_wet.pipe(merge, df_dry)
    df_imaged = df_label.pipe(drop_no_image, sample_ids)
    return df_imaged


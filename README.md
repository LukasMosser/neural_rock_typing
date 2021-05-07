# Neural Rock Typing
Can machines help us understand how to distinguish rock types?

### Authors

Gregor Baechle, George Ghon, Lukas Mosser
_Carbonate Complexities Group_, 2020

## Colab Training

To load a notebook for training a model in Google Colab, follow this link:  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LukasMosser/neural_rock_typing/]

## Install Environment

Install the provided conda environment:
```bash
conda env create -f environment.yml
```

# Training Dataset

Find and download dataset here:

[Gdrive](https://drive.google.com/drive/folders/1_xBydGIVzWQe9htU3Yacqa34h2vEGoE5?usp=sharing)

based on: [Digital Rocks Portal](https://www.digitalrocksportal.org/projects/215)

## Model Checkpoint

Download the pretrained model here:
[Gdrive](https://drive.google.com/drive/folders/1vtct_onMmL2Ax13hMILwJRoGDG_GMDev?usp=sharing)

## Running the application

To run the application, open a terminal and execute following commands:
```bash
streamlit run apps/viewer.py
```

This will open up a new browser window with a live application view.



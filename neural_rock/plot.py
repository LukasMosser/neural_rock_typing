import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from neural_rock.utils import MEAN_TRAIN, STD_TRAIN


def visualize_batch(loader: DataLoader):
    """
    Plots a batch of images in the right color scaling.
    """
    fig, axarr = plt.subplots(4, 4, figsize=(12, 12))
    for dat, _ in loader:
        break
    for ax, im in zip(axarr.flatten(), dat.numpy()):
        im = im.transpose(1, 2, 0)*STD_TRAIN+MEAN_TRAIN
        ax.imshow(im)
    plt.show()
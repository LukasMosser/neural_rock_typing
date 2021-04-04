import numpy as np
import matplotlib.pyplot as plt


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def visualize_batch(loader):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    fig, axarr = plt.subplots(4, 4, figsize=(12, 12))
    for dat, _ in loader:
        break
    for ax, im in zip(axarr.flatten(), dat.numpy()):
        im = im.transpose(1, 2, 0)*std+mean
        ax.imshow(im)
    plt.show()
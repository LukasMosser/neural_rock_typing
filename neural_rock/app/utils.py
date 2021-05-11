import numpy as np
import torch
from torch.autograd import Variable
import albumentations as A


def make_cam_map(X, grad_cam, i, device='cpu', ratio=224.0/512.0):
    """
    Computes a CAM map for a given image.
    Assumes a certain ratio between input training image size and test image size rescaling.
    This is to maintain the scale of pixels used during training and testing.
    """
    transform = A.Compose([
        A.Resize(int(ratio * X.shape[0]), int(ratio * X.shape[1])),
        A.Normalize()])

    X = transform(image=X)['image'].transpose(2, 0, 1)
    X = Variable(torch.from_numpy(X).unsqueeze(0), requires_grad=True)
    X = X.to(device)
    cam_map = compute_cam(X, grad_cam, i, device=device)

    with torch.no_grad():
        y_prob = torch.nn.functional.softmax(grad_cam.model(X), dim=1)

    return cam_map, y_prob


def compute_cam(X, grad_cam, i, device="cpu"):
    """
    Computes the CAM map and min-max rescales it.
    """
    gb = grad_cam(X, i)
    gb = (gb - np.min(gb)) / (np.max(gb) - np.min(gb))
    return gb
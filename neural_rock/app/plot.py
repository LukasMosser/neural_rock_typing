import numpy as np
import xarray as xa
import holoviews as hv
from datashader.utils import ngjit
import datashader as ds
hv.extension('bokeh')


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
    """
    Recombines colorbands
    """
    xs, ys = r['y'], r['x']
    r, g, b = [ds.utils.orient_array(im) for im in (r, g, b)]
    return hv.RGB((xs, ys[::-1], r, g, b), vdims=list('RGB'))


def create_holoviews_cam(cam):
    """
    Create holoviews object of thin-section image.
    """
    coords = {'x': np.arange(cam.shape[0]), 'y': np.arange(cam.shape[1])}

    cam = xa.DataArray(name='cam', data=cam, coords=coords, dims=['x', 'y'])
    regridded_cam = hv.Image(cam, kdims=['x', 'y'], vdims=hv.Dimension('z', range=(0, 1)))
    return regridded_cam


def create_holoviews_thinsection(image_patch):
    """
    Create Holoviews compatible xarray for RGB image
    """
    coords = {'x': np.arange(image_patch.shape[0]), 'y': np.arange(image_patch.shape[1])}

    r_ = xa.DataArray(name='r', data=image_patch[..., 0], coords=coords, dims=['x', 'y'])
    g_ = xa.DataArray(name='g', data=image_patch[..., 1], coords=coords, dims=['x', 'y'])
    b_ = xa.DataArray(name='b', data=image_patch[..., 2], coords=coords, dims=['x', 'y'])

    rgb = combine_bands(r_, g_, b_)

    return rgb

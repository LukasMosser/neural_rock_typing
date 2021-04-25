import numpy as np
import holoviews as hv
import panel as pn

box = hv.BoxWhisker((np.random.randint(0, 10, 100), np.random.randn(100)), 'Group').sort()

hv_layout = pn.panel(pn.Row(box))
hv_layout.pprint()


"""import numpy as np
from torchvision import transforms
import torch
from torch.autograd import Variable
from neural_rock.cam import GradCam
import albumentations as A
from neural_rock.app.utils import load_model, load_data, get_train_test_split, Model, compute_images, MEAN_TRAIN, STD_TRAIN
from neural_rock.app.plot import create_holoviews_cam, create_holoviews_thinsection
from imageio import imread
import holoviews as hv
import panel as pn
from neural_rock.model import make_vgg11_model
hv.extension('bokeh')
from bokeh.plotting import show

train_test_split = get_train_test_split()

labelset = 'Dunham'
device = 'cpu'

image_paths, modified_label_map, image_names, class_names = load_data(labelset)

for id, img in image_paths.items():
    if id in train_test_split['train']:
        img['train'] = 'Train'
    else:
        img['train'] = 'Test'

if labelset == "Lucia":
    chkpt = "./data/models/Lucia/v1/epoch=29-step=629.ckpt"
elif labelset == "DominantPore":
    chkpt = "./data/models/DominantPore/v1/epoch=0-step=20.ckpt"
elif labelset == "Dunham":
    chkpt = "./data/models/Dunham/v1/epoch=11-step=251.ckpt"
#model = load_model(chkpt, num_classes=len(class_names))

image_name_map = {}
image_names = []
for image_name, dset in image_paths.items():
    image_n = "-".join([str(image_name), dset['label'], str(dset['train'])])
    image_names.append(image_n)
    image_name_map[image_n] = image_name

image_id = image_names[0]

image_id = image_name_map[image_id]

min_layer = 0
#max_layer = len(list(model.feature_extractor.modules()))-2
#layer = len(list(model.feature_extractor.modules()))-2

class_n = class_names[0]
class_idx = np.argwhere(np.array(class_names) == class_n)[0, 0]

#model = Model(model.feature_extractor, model.classifier)
#model.to(device)
#grad_cam = GradCam(model=model.to(device),
#                   feature_module=model.feature_extractor.to(device),
#                   target_layer_names=[str(layer)],
#                   device=device)

X_np = imread(image_paths[image_id]['path'])
mask = imread(image_paths[image_id]['mask_path'])
mask = (1 - (mask > 0).astype(np.uint8))
X_np = X_np * np.repeat(np.expand_dims(mask, axis=-1), 3, axis=-1)

label = image_paths[image_id]['label']

ratio = 224.0/512

transform = A.Compose([
    A.Resize(int(ratio*X_np.shape[0]), int(ratio*X_np.shape[1])),
    A.Normalize()])

resize = transforms.Resize((X_np.shape[0], X_np.shape[1]))

X = transform(image=X_np)['image'].transpose(2, 0, 1)

#X = Variable(torch.from_numpy(X).unsqueeze(0), requires_grad=True).to(device)

#with torch.no_grad():
#    output = model(X).to(device)

image_patch = X_np#np.transpose(X_np, (1, 2, 0)) * STD_TRAIN + MEAN_TRAIN
#X = X.to(device)
#maps = compute_images(X, grad_cam, len(class_names), resize=resize, device=device)

#hv_cam = create_holoviews_cam(maps[0].T)
hv_thinsection = create_holoviews_thinsection(image_patch)

#hv_cam = hv_cam.opts(cmap="inferno", alpha=0.5)
#hv_thinsection = hv_thinsection.opts(responsive=False, width=900, height=900)
print("gonna show")
occupancy = pn.Row((hv_thinsection).opts(width=600, active_tools=['xwheel_zoom', 'pan']))
occupancy"""
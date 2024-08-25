import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "7"

import numpy as np
import torch
import tqdm
import pandas as pd
from utils.help_utils import cpu, load_yaml
import tifffile as tif
from spline_psf.calibration_io import SMAPSplineCoefficient
from utils.dataGenerator import DataGenerator

# this program has been tested by inference metrics.

yaml_file = 'param_astig.yaml'
params = load_yaml(yaml_file)

# model_path = "/home/feiyue/LiteLoc_spline/training_model/liteloc_spline_astig/checkpoint.pkl"
# liteloc = torch.load(model_path)

DataGen = DataGenerator(params.Training, params.Camera, params.PSF_model)

activation = pd.read_csv("/home/feiyue/LiteLoc_spline/pos4w5_frame5w4_size32_activations.csv")
activation = activation.sort_values(by='frame', ascending=True)
frame_ix = torch.tensor(activation.iloc[:, 1])
x = torch.unsqueeze(torch.tensor(activation.iloc[:, 3] / 108), dim=1)
y = torch.unsqueeze(torch.tensor(activation.iloc[:, 2] / 108), dim=1)
z = torch.unsqueeze(torch.tensor(activation.iloc[:, 4]), dim=1)
intensity = torch.tensor(activation.iloc[:, 5])

xyz_px = torch.cat([x, y, z], dim=1)

image_size = 32

DataGen.train_size_x, DataGen.train_size_y = image_size, image_size
DataGen.psf = SMAPSplineCoefficient(calib_file=DataGen.spline_params.calibration_file).init_spline(
    xextent=[-0.5, image_size-0.5],
    yextent=[-0.5, image_size-0.5],
    img_shape=[image_size, image_size],
    device=DataGen.spline_params.device_simulation,
    roi_size=None,
    roi_auto_center=None
    ).cuda()

unique_elements, counts = torch.unique(frame_ix, return_counts=True)
batch_size = 2000
molecule_ind = 0
img_sim_tmp = [[] for j in range(int(1e8))]
image_ind = 0

frame_emitter = 0
for j in tqdm.tqdm(range(frame_ix.max())):
    if j == unique_elements[frame_emitter].item() - 1:
        molecule_count = counts[frame_emitter]
        frame_ix_tmp = frame_ix[molecule_ind:molecule_ind + molecule_count]
        xyz_px_tmp = xyz_px[molecule_ind:molecule_ind + molecule_count]
        intensity_tmp = intensity[molecule_ind:molecule_ind + molecule_count]
        img_sim_tmp[j] = cpu(DataGen.simulated_splinePSF_from_gt(xyz_px_tmp, intensity_tmp, frame_ix_tmp)).astype(
            np.uint16)[0, 0]
        frame_emitter = frame_emitter + 1
        molecule_ind = molecule_ind + molecule_count
    else:
        img_sim_tmp[j] = cpu(torch.ones([image_size, image_size]) * 20).astype(np.uint16)

img_sim_save = np.array(img_sim_tmp).ravel().tolist()


# for i in range(int(np.ceil(unique_elements.shape[0] / batch_size))):
#     for frame in unique_elements:
#     molecule_count = counts[i*batch_size:(i+1)*batch_size].sum()
#     frame_ix_tmp = frame_ix[molecule_ind:molecule_ind + molecule_count]
#     xyz_px_tmp = xyz_px[molecule_ind:molecule_ind + molecule_count]
#     intensity_tmp = intensity[molecule_ind:molecule_ind + molecule_count]
#     img_sim_tmp[i] = cpu(DataGen.simulated_splinePSF_from_gt(xyz_px_tmp, intensity_tmp, frame_ix_tmp)).astype(np.uint16)[:, 0]
#
#     molecule_ind = molecule_ind + molecule_count
#     print("already simulate " + str((i+1)*batch_size) + " frames")
# img_sim_save = [image for image_list in img_sim_tmp for image in image_list]

tif.imwrite("/home/feiyue/LiteLoc_spline/pos4w5_frame5w4_size32.tif", img_sim_save, dtype=np.uint16)

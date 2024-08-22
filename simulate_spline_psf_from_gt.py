import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "7"

import numpy as np
import torch
import pandas as pd
from utils.help_utils import cpu, load_yaml
import tifffile as tif
from spline_psf.calibration_io import SMAPSplineCoefficient
import matplotlib.pyplot as plt

from utils.dataGenerator import DataGenerator

yaml_file = 'param_tetra6.yaml'
params = load_yaml(yaml_file)

# model_path = "/home/feiyue/LiteLoc_spline/training_model/liteloc_spline_astig/checkpoint.pkl"
# liteloc = torch.load(model_path)

DataGen = DataGenerator(params.Training, params.Camera, params.PSF_model)

activation = pd.read_csv("/home/feiyue/LiteLoc_local_torchsimu/random_points/Tetrapod_6um/hsnr_ld/activations.csv")
frame_ix = torch.tensor(activation.iloc[:, 1])
x = torch.unsqueeze(torch.tensor(activation.iloc[:, 3] / 108), dim=1)
y = torch.unsqueeze(torch.tensor(activation.iloc[:, 2] / 108), dim=1)
z = torch.unsqueeze(torch.tensor(activation.iloc[:, 4]), dim=1)
intensity = torch.tensor(activation.iloc[:, 5])

xyz_px = torch.cat([x, y, z], dim=1)

# liteloc.DataGen.train_size_x, liteloc.DataGen.train_size_y = 32, 32
# liteloc.DataGen.psf = SMAPSplineCoefficient(calib_file=liteloc.DataGen.spline_params.calibration_file).init_spline(
#     xextent=[-0.5, 31.5],
#     yextent=[-0.5, 31.5],
#     img_shape=[32, 32],
#     device=liteloc.DataGen.spline_params.device_simulation,
#     roi_size=None,
#     roi_auto_center=None
#     ).cuda()

# img_sim = cpu(liteloc.DataGen.simulated_splinePSF_from_gt(xyz_px, intensity, frame_ix)).astype(np.uint16)
img_sim = cpu(DataGen.simulated_splinePSF_from_gt(xyz_px, intensity, frame_ix)).astype(np.uint16)

img_sim_save = [img_sim[i] for i in range(img_sim.shape[0])]

tif.imwrite("/home/feiyue/LiteLoc_spline/spline_data/hsnr_ld_spline_tetra6_new.tif", img_sim_save, dtype=np.uint16)

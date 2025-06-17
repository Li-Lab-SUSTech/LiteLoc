import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "5"  # If certain GPUs will be used, please set the index. Otherwise, delete this line.

import logging
logger = logging.getLogger()
logger.setLevel(logging.ERROR)

import torch
import numpy as np
# import time
import argparse
# from network import multi_process
from utils.help_utils import dict2device, cpu #, load_yaml_infer
from network.loc_model import LocModel
# from utils.visual_utils import show_sample_psf, show_train_img
import matplotlib.pyplot as plt

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--infer_params_dict', type=str)
    parser.add_argument('--device', type=str, default='cpu')
    # parser.add_argument('--image_pth', type=str)
    args = parser.parse_args()

    params_dict = torch.load(args.infer_params_dict, map_location = args.device)
    
    if args.device ==  'cuda':
        assert torch.cuda.is_available()
    
    liteloc = LocModel(params_dict['Parameters'], args.device)
    
    state_dict = dict2device(params_dict['Model_state'], args.device)
    
    liteloc.network.load_state_dict(state_dict)
    
    liteloc.network.eval()
        
    with torch.no_grad():
        
        xemit, yemit, z, S, Nphotons, s_mask, gt, img_sim = next(iter(liteloc.valid_data))
        molecule_array = liteloc.analyze(img_sim)
        for i in range(img_sim.shape[1]):
            plt.imshow(np.squeeze(cpu(img_sim[0, i, 0])))
            plt.colorbar()
            plt.show()
            w_index = torch.where(molecule_array[:, 0] == i)
            print("Ground Truth:")
            
            print(gt[0, i, : s_mask[0,i].sum().int()])
            
            print("Inference Result:")
            print(molecule_array[w_index][1 : -1])
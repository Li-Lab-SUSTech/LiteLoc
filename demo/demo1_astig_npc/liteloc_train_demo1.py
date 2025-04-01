import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "5"

import argparse
from utils.help_utils import load_yaml_train, writelog, setup_seed
from utils.visual_utils import show_sample_psf, show_train_img
from network.loc_model import LocModel

if __name__ == '__main__':

    setup_seed(15)

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_params_path', type=str, default='train_params_demo1.yaml')
    args = parser.parse_args()

    params = load_yaml_train(args.train_params_path)

    liteloc = LocModel(params)

    show_sample_psf(psf_pars=params.PSF_model)
    show_train_img(image_num=4, camera_params=params.Camera, psf_params=params.PSF_model, train_params=params.Training)

    writelog(params.Training.result_path)

    liteloc.train()
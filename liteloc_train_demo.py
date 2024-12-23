import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "3"

import argparse
from utils.help_utils import load_yaml_fy, writelog, setup_seed
from utils.visual_utils import show_sample_psf, show_train_img
from network.loc_model import LitelocModel

if __name__ == '__main__':

    setup_seed(15)

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_params_path', type=str, default='train_params_demo_fig3a.yaml')
    args = parser.parse_args()

    params = load_yaml_fy(args.train_params_path)
    params.Training.project_path = os.path.join(os.path.expanduser('~'), params.Training.project_name)

    liteloc = LitelocModel(params)

    show_sample_psf(psf_pars=params.PSF_model, train_pars=params.Training)
    show_train_img(image_num=4, camera_params=params.Camera, psf_params=params.PSF_model, train_params=params.Training)

    writelog(params.Training.project_path + params.Training.result_path)

    liteloc.train()
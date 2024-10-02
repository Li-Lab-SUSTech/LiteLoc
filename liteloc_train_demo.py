import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "6"  # If certain GPUs will be used, please set the index. Otherwise, delete this line.

from utils.help_utils import load_yaml, writelog, setup_seed
from utils.visual_utils import show_sample_psf, show_train_img
from network.loc_model import LitelocModel

if __name__ == '__main__':

    setup_seed(15)

    yaml_file = 'train_params_demo_fig3d.yaml'
    params = load_yaml(yaml_file)

    liteloc = LitelocModel(params)

    show_sample_psf(psf_pars=params.PSF_model, train_pars=params.Training)
    show_train_img(image_num=4, camera_params=params.Camera, psf_params=params.PSF_model, train_params=params.Training)

    writelog(params.Training.result_path)

    liteloc.train_spline()
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "5"

from utils.help_utils import load_yaml, writelog, setup_seed
from utils.visual_utils import show_sample_psf, show_train_img
from network.loc_model_decode import DECODEModel

if __name__ == '__main__':

    setup_seed(15)

    yaml_file = 'param_decode_tetra6_train.yaml'
    params = load_yaml(yaml_file)

    writelog(params.Training.result_path)

    decode = DECODEModel(params)

    show_sample_psf(psf_pars=params.PSF_model, train_pars=params.Training)
    show_train_img(image_num=4, camera_params=params.Camera, psf_params=params.PSF_model, train_params=params.Training)


    decode.train_spline()
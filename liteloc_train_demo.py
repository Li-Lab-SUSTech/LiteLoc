import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "3"

import argparse
from utils.help_utils import load_yaml_train, writelog, setup_seed
from utils.visual_utils import show_sample_psf, show_train_img


if __name__ == '__main__':

    setup_seed(15)

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', type = str, default = 'LiteLoc', choices = ['LiteLoc', 'DECODE'])
    parser.add_argument('-p', '--train_params_path', type=str, default='train_params_demo_fig3a.yaml')
    args = parser.parse_args()
    
    # if args.model_name == 'DECODE':
    #     os.environ['CUDA_VISIBLE_DEVICES'] = "1"

    params = load_yaml_train(args.train_params_path)
    
    if args.model_name == 'LiteLoc':
        from network.loc_model import LitelocModel
        model = LitelocModel(params)
    else:
        from network.loc_model_decode import DECODEModel
        model = DECODEModel(params)

    # liteloc = LitelocModel(params)

    show_sample_psf(psf_pars=params.PSF_model)
    show_train_img(image_num=4, camera_params=params.Camera, psf_params=params.PSF_model, train_params=params.Training)

    writelog(params.Training.result_path)

    # liteloc.train()
    model.train()
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = "1"  # For Apple Silicon users, enable MPS fallback

from utils.help_utils import load_yaml_train, writelog, setup_seed
from utils.visual_utils import show_sample_psf, show_train_img
from network.loc_model_decode import LocModel

if __name__ == '__main__':

    setup_seed(15)

    yaml_file = 'param_decode_train_m1_oil_astig_npc.yaml'
    params = load_yaml_train(yaml_file)
    decode = LocModel(params)

    writelog(params.Training.result_path)

    print(params)

    show_sample_psf(psf_pars=params.PSF_model)
    show_train_img(image_num=4, camera_params=params.Camera, psf_params=params.PSF_model, train_params=params.Training)

    decode.train()
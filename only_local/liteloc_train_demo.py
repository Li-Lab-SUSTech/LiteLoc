import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "6"

from utils.help_utils import load_yaml, writelog, setup_seed
from utils.visual_utils import show_sample_psf
from network.loc_model import LitelocModel

if __name__ == '__main__':

    setup_seed(15)

    yaml_file = 'param_astig_new_ui.yaml'
    params = load_yaml(yaml_file)

    liteloc = LitelocModel(params)

    show_sample_psf(psf_pars=params.PSF_model, train_pars=params.Training)

    writelog(params.Training.result_path)

    liteloc.train_spline()
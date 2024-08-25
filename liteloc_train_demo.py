import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "6"

from utils.help_utils import load_yaml, writelog, setup_seed
from utils.visual_utils import ShowSamplePSF
from network.loc_model import LitelocModel

if __name__ == '__main__':

    setup_seed(15)

    yaml_file = 'param_astig.yaml'
    params = load_yaml(yaml_file)

    liteloc = LitelocModel(params)

    ShowSamplePSF(psf_pars=params.PSF_model, train_pars=params.Training)

    writelog(params.Training.result_path)

    liteloc.train_spline()
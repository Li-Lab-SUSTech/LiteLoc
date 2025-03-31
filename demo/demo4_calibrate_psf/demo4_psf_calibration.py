from utils.vectorpsf_fit import beads_psf_calibrate
from utils.help_utils import load_yaml


calib_params = load_yaml("params_demo4_psf_calibration.yaml")

beads_psf_calibrate(calib_params)
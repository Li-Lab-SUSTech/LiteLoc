import scipy.io as sio
import torch
import numpy as np

import spline_psf.psf_kernel as psf_kernel
from utils.help_utils import load_h5


class SMAPSplineCoefficient:
    """Wrapper class as an interface for MATLAB Spline calibration data."""
    def __init__(self, calib_file):
        """
        Loads a calibration file from SMAP and the relevant meta information
        Args:
            file:
        """
        self.calib_file = calib_file
        if calib_file.split('.')[-1] == 'mat':
            calib_mat = sio.loadmat(self.calib_file, struct_as_record=False, squeeze_me=True)
            if 'cspline_psf_model' in calib_mat.keys():
                self.calib_mat = calib_mat['cspline_psf_model']
            else:
                self.calib_mat = calib_mat['SXY'].cspline
            self.coeff = torch.from_numpy(self.calib_mat.coeff)
            self.ref0 = (self.calib_mat.x0 - 1, self.calib_mat.x0 - 1, self.calib_mat.z0)
            self.dz = self.calib_mat.dz
            self.spline_roi_shape = self.coeff.shape[:3]
        elif calib_file.split('.')[-1] == 'h5':
            calib_dict, params = load_h5(calib_file)
            self.coeff = torch.from_numpy(np.ascontiguousarray(np.flip(np.transpose(calib_dict['locres']['coeff'], (3, 2, 1, 0)), axis=2)))
            self.ref0 = (self.coeff.shape[0]//2 + 1, self.coeff.shape[1]//2 + 1, self.coeff.shape[2]//2 + 1)
            self.dz = params['pixel_size']['z']*1000  # in nm
            self.spline_roi_shape = self.coeff.shape[:3]
        else:
            raise ValueError('Unsupported calibration file format. Use .mat or .h5 files.')

    def init_spline(self, xextent, yextent, img_shape, device='cuda:1' if torch.cuda.is_available() else 'cpu', **kwargs):
        """
        Initializes the CubicSpline function

        Args:
            xextent:
            yextent:
            img_shape:
            device: on which device to simulate

        Returns:

        """
        psf = psf_kernel.CubicSplinePSF(xextent=xextent, yextent=yextent, img_shape=img_shape, ref0=self.ref0,
                                        coeff=self.coeff, vx_size=(1., 1., self.dz), device=device, **kwargs)

        return psf
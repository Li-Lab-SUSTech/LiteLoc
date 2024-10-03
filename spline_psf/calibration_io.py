import scipy.io as sio
import torch

import spline_psf.psf_kernel as psf_kernel


class SMAPSplineCoefficient:
    """Wrapper class as an interface for MATLAB Spline calibration data."""
    def __init__(self, calib_file):
        """
        Loads a calibration file from SMAP and the relevant meta information
        Args:
            file:
        """
        self.calib_file = calib_file
        calib_mat = sio.loadmat(self.calib_file, struct_as_record=False, squeeze_me=True)
        if 'cspline_psf_model' in calib_mat.keys():
            self.calib_mat = calib_mat['cspline_psf_model']
        else:
            self.calib_mat = calib_mat['SXY'].cspline
        self.coeff = torch.from_numpy(self.calib_mat.coeff)
        self.ref0 = (self.calib_mat.x0 - 1, self.calib_mat.x0 - 1, self.calib_mat.z0)
        self.dz = self.calib_mat.dz
        self.spline_roi_shape = self.coeff.shape[:3]

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
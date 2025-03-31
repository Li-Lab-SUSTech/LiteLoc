from abc import ABC, abstractmethod  # abstract class
import torch
from utils.help_utils import gpu


class Camera(ABC):
    """
    Abstract camera class. All children must implement a forward/backward method.
    """

    def __init__(self, camera_params):
        self.qe = camera_params['qe']
        self.spurious_charge = camera_params['spurious_charge']
        self.read_noise_sigma = camera_params['read_noise_sigma']
        self.e_per_adu = camera_params['e_per_adu']
        self.baseline = camera_params['baseline']
        self.em_gain = None

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def backward(self, x_adu):
        """
        Calculates the expected number of photons from a camera image.

        Args:
            x_adu (torch.Tensor): input in ADU

        Returns:
            torch.Tensor: expected photon image
        """

        x_e = (x_adu - self.baseline) * self.e_per_adu
        if self.em_gain is not None:
            x_e /= self.em_gain
        x_e -= self.spurious_charge
        x_photon = torch.clamp(x_e / self.qe, min=1e-10)

        return x_photon

    def forward_wo_noise(self, x_photon):
        x_e = x_photon * self.qe + self.spurious_charge
        if self.em_gain is not None:
            x_e *= self.em_gain
        x_adu = x_e / self.e_per_adu + self.baseline
        return x_adu


class EMCCD(Camera):
    """
    An EMCCD class with unique parameter em_gain.
    """

    def __init__(self, camera_params):
        """

        Args:
            camera_params (dict): dict containing the camera parameters.
                qe: quantum efficiency (0-1);
                spurious_charge: manufacturer-quoted spurious charge; clock-induced charge only, dark counts negligible;
                e_per_adu: analog-to-digital conversion factor;
                baseline: manufacturer basiline or offset;
                read_noise_sigma: std of the readout noise (gaussian distribution), a single value;
                em_gain: electron-multiplying gain, for EMCCD only;

        Returns:

        """

        super().__init__(camera_params)
        self.em_gain = camera_params['em_gain']

    def forward(self, x_photon):
        """
        Simulates the EMCCD camera effect on the input image.

        Args:
            x_photon (torch.Tensor): input in photons

        Returns:
            torch.Tensor: simulated camera image in ADU
        """

        x_poisson = torch.distributions.Poisson(x_photon * self.qe + self.spurious_charge).sample()

        x_gamma = torch.distributions.Gamma(x_poisson + torch.finfo(x_poisson.dtype).eps, 1 / self.em_gain).sample()

        read_noise_map = torch.zeros_like(x_gamma)+self.read_noise_sigma
        x_read = x_gamma + read_noise_map * torch.randn_like(x_gamma)

        x_adu = torch.clamp(x_read / self.e_per_adu + self.baseline, min=1)

        return x_adu


class SCMOS(Camera):
    """
    An sCMOS class, the read noise should be a spatially variant map cover the full-frame pixels
    """

    def __init__(self, camera_params):
        super().__init__(camera_params)
        try:
            self.read_noise_sigma = camera_params['read_noise_sigma']
        except KeyError:
            self.read_noise_sigma = None
        try:
            self.read_noise_map = camera_params['read_noise_map']
        except KeyError:
            self.read_noise_map = None
        if (self.read_noise_map is None) != (self.read_noise_sigma is None):
            pass
        else:
            raise ValueError('you must define either read_noise_sigma xor read_noise_map')

    def forward(self, x_photon, fov_xy=None):
        """
        Simulates the sCMOS camera effect on the input image.

        Args:
            x_photon (torch.Tensor): input in photons
            fov_xy (list or None): when using a read noise map and simulate in a divide and conquer strategy,
                the input image is only a sub-fov, thus needs the corresponding sub-fov to index the
                read noise map. If using a single value read noise sigma, this parameter is not needed.

        Returns:
            torch.Tensor: simulated camera image in ADU
        """

        x_poisson = torch.distributions.Poisson(x_photon * self.qe + self.spurious_charge).sample()

        if self.read_noise_map is None:
            curr_read_noise_map = None
        else:
            curr_read_noise_map = gpu(self.read_noise_map[fov_xy[2]:fov_xy[3] + 1, fov_xy[0]:fov_xy[1] + 1])

        if curr_read_noise_map is None:
            assert self.read_noise_sigma is not None, \
                "both read_noise_sigma and read noise map are None, please define either"
            curr_read_noise_sigma = torch.zeros_like(x_poisson)+self.read_noise_sigma
        else:
            curr_read_noise_sigma = torch.zeros_like(x_poisson)+curr_read_noise_map

        x_read = x_poisson + curr_read_noise_sigma * torch.randn_like(x_poisson)

        x_adu = torch.clamp(x_read / self.e_per_adu + self.baseline, min=1)

        return x_adu


class IdeaCamera(SCMOS):
    """
    An idea camera only consider shot noise.
    """

    def __init__(self):
        camera_params = {'qe': 1.0, 'spurious_charge': 0.0,
                         'read_noise_sigma': 0.0, 'read_noise_map': None,
                         'e_per_adu': 1.0, 'baseline': 0.0}
        super().__init__(camera_params)


def instantiate_camera(camera_params_dict):
    """
    Instantiate a camera object.
    """

    if camera_params_dict['camera_type'].upper() == 'EMCCD':
        camera = EMCCD(camera_params_dict)
    elif camera_params_dict['camera_type'].upper() == 'SCMOS':
        camera = SCMOS(camera_params_dict)
    elif camera_params_dict['camera_type'].upper() == 'IDEA':
        camera = IdeaCamera()
    else:
        raise NotImplementedError('Camera type not supported.')
    return camera


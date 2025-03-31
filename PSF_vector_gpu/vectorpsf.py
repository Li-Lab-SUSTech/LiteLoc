import ctypes
import sys
from abc import ABC, abstractmethod  # abstract class
import torch
import torch.nn as nn
import numpy as np
from utils.help_utils import gpu

from deprecated import deprecated

class VectorPSF(ABC):
    """
    Abstract base class for vector psf simulators
    """

    @abstractmethod
    def simulate(self, *args, **kwargs):
        """
        run the simulation
        """

        raise NotImplementedError

    @staticmethod
    def _gauss2D_kernel(shape=(3, 3), sigmax=0.5, sigmay=0.5, data_type=torch.float32):
        """
        2D gaussian mask for VectorPSF otf rescale
        """

        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = torch.meshgrid(gpu(torch.arange(-m, m + 1, 1), data_type=data_type),
                              gpu(torch.arange(-n, n + 1, 1), data_type=data_type), indexing='ij')
        h = torch.exp(-(x * x) / (2. * sigmax * sigmax + 1e-6) - (y * y) / (2. * sigmay * sigmay + 1e-6))
        torch.clamp(h, min=0)
        # h[h < torch.finfo(h.dtype).eps * h.max()] = 0
        with torch.no_grad():
            sumh = h.sum()
        output = h / sumh if sumh != 0 else h
        return output

    @staticmethod
    def otf_rescale(psfdata, sigma_xy):
        """
        Convolve the psf with a gaussian kernel, namely otf rescale

        Args:
            psfdata (torch.Tensor): psf data to be blured
            sigma_xy (torch.Tensor): sigma x y of the gaussian kernel, unit pixel

        Returns:
            torch.Tensor: blured psf data
        """

        kernel = VectorPSF._gauss2D_kernel(shape=(5, 5),
                                           sigmax=sigma_xy[0],
                                           sigmay=sigma_xy[1],
                                           data_type=psfdata.dtype).reshape((1, 1, 5, 5))
        psf_size = psfdata.shape[1]
        psfdata = psfdata.view(-1, 1, psf_size, psf_size)
        tmp = nn.functional.conv2d(psfdata, kernel, padding=2, stride=1)
        outdata = tmp.view(-1, psf_size, psf_size)

        return outdata


class VectorPSFTorch(VectorPSF):
    """
    Vector psf simulator using pytorch, thus psf parameters can be optimized by pytorch
    """

    def __init__(self, psf_params, zernike_aber, objstage0, req_grad=False, data_type=torch.float64):
        """

        Args:
            psf_params (dict): PSF parameters for simulation except emitter positions
            req_grad (bool): whether the PSF parameters are required gradient
            data_type (torch.dtype): data type for the PSF parameters

        Returns:
            VectorPSFTorch: an instance of VectorPSFTorch
        """

        self.data_type = data_type
        if data_type == torch.float64:
            self.complex_type = torch.complex128
        elif data_type == torch.float32:
            self.complex_type = torch.complex64
        elif data_type == torch.float16:
            self.complex_type = torch.complex32
        else:
            raise ValueError(f'unsupported data type {data_type}')

        zernike_mode = zernike_aber[:, 0:2]
        zernike_coef = zernike_aber[:, 2]

        self.na = torch.tensor(psf_params.NA, device='cuda', dtype=self.data_type)
        self.wavelength = torch.tensor(psf_params.wavelength, device='cuda', dtype=self.data_type)
        self.refmed = torch.tensor(psf_params.refmed, device='cuda', dtype=self.data_type, requires_grad=req_grad)
        self.refcov = torch.tensor(psf_params.refcov, device='cuda', dtype=self.data_type, requires_grad=req_grad)
        self.refimm = torch.tensor(psf_params.refimm, device='cuda', dtype=self.data_type, requires_grad=req_grad)
        self.zernike_mode = torch.tensor(zernike_mode, device='cuda', dtype=self.data_type)
        self.zernike_coef = torch.tensor(zernike_coef, device='cuda', dtype=self.data_type,
                                         requires_grad=req_grad)
        self.zernike_coef_map = None
        self.objstage0 = torch.tensor(objstage0, device='cuda', dtype=self.data_type,
                                      requires_grad=req_grad)
        # try:
        #     self.zemit0 = torch.tensor(zemit0, device='cuda', dtype=self.data_type, requires_grad=req_grad)
        # except KeyError:
        self.zemit0 = torch.tensor(-objstage0/psf_params.refimm * psf_params.refmed,
                                   device='cuda',
                                   dtype=self.data_type,
                                   requires_grad=req_grad)

        self.pixel_size_xy = torch.tensor([psf_params.pixelSizeX, psf_params.pixelSizeY], device='cuda', dtype=self.data_type)
        self.otf_rescale_xy = torch.tensor([psf_params.psfrescale, psf_params.psfrescale], device='cuda', dtype=self.data_type,
                                           requires_grad=req_grad)
        self.npupil = psf_params.Npupil
        self.psf_size = psf_params.psfSizeX

        self._pre_compute()

    @staticmethod
    def get_zernike(orders, xpupil, ypupil):
        """
        Calculate zernike polynomials on pupil plane

        Args:
            orders (torch.Tensor): zernike orders, 2D array, first column is radial order,
                second column is azimuthal order
            xpupil (torch.Tensor): x coordinate of pupil plane
            ypupil (torch.Tensor): y coordinate of pupil plane

        Returns:
            torch.Tensor: zernike polynomials
        """

        xpupil = torch.real(xpupil)
        ypupil = torch.real(ypupil)
        zersize = orders.shape
        Nzer = zersize[0]
        radormax = int(max(orders[:, 0]))
        azormax = int(max(abs(orders[:, 1])))
        [Nx, Ny] = xpupil.shape

        # zerpol = np.zeros( [radormax+1,azormax+1,Nx,Ny] )
        zerpol = torch.zeros([21, 6, Nx, Ny], device='cuda')
        rhosq = xpupil ** 2 + ypupil ** 2
        rho = torch.sqrt(rhosq)
        zerpol[0, 0, :, :] = torch.ones_like(xpupil)

        for jm in range(1, azormax + 2 + 1):
            m = jm - 1
            if m > 0:
                zerpol[jm - 1, jm - 1, :, :] = rho * torch.squeeze(zerpol[jm - 1 - 1, jm - 1 - 1, :, :])

            zerpol[jm + 2 - 1, jm - 1, :, :] = ((m + 2) * rhosq - m - 1) * torch.squeeze(zerpol[jm - 1, jm - 1, :, :])
            for p in range(2, radormax - m + 2 + 1):
                n = m + 2 * p
                jn = n + 1
                zerpol[jn - 1, jm - 1, :, :] = (2 * (n - 1) * (n * (n - 2) * (2 * rhosq - 1) - m ** 2) * torch.squeeze(
                    zerpol[jn - 2 - 1, jm - 1, :, :]) -
                                                n * (n + m - 2) * (n - m - 2) * torch.squeeze(
                            zerpol[jn - 4 - 1, jm - 1, :, :])) / ((n - 2) * (n + m) * (n - m))

        phi = torch.atan2(ypupil, xpupil)
        allzernikes = torch.zeros([Nzer, Nx, Ny], device='cuda')
        for j in range(1, Nzer + 1):
            n = int(orders[j - 1, 0])
            m = int(orders[j - 1, 1])
            if m >= 0:
                allzernikes[j - 1, :, :] = torch.squeeze(zerpol[n + 1 - 1, m + 1 - 1, :, :]) * torch.cos(m * phi)
            else:
                allzernikes[j - 1, :, :] = torch.squeeze(zerpol[n + 1 - 1, -m + 1 - 1, :, :]) * torch.sin(-m * phi)

        # plt.figure(constrained_layout=True)
        # for i in range(21):
        #     plt.subplot(3, 7, i + 1)
        #     plt.imshow(cpu(allzernikes[i]))
        # plt.show()

        return allzernikes

    # @staticmethod
    def czt(self, datain, A, B, D):
        """
        Execute the chirp-z transform

        Args:
            datain (torch.Tensor): input data, 2D array
            A (torch.Tensor): chirp parameter, 1D array
            B (torch.Tensor): chirp parameter, 1D array
            D (torch.Tensor): chirp parameter, 1D array

        Returns:
            torch.Tensor: output data, 2D array
        """

        N = A.shape[1]
        M = B.shape[1]
        L = D.shape[1]
        K = datain.shape[0]

        # torch.repeat_interleave is too slow
        # t0 = time.time()
        # Amt = torch.repeat_interleave(A, K, 0)
        # Bmt = torch.repeat_interleave(B, K, 0)
        # Dmt = torch.repeat_interleave(D, K, 0)
        # print('torch czt: ', time.time() - t0)
        Amt = A.expand(K, N)
        Bmt = B.expand(K, M)
        Dmt = D.expand(K, L)

        cztin = torch.zeros([K, L], dtype=self.complex_type, device='cuda')
        cztin[:, 0:N] = Amt * datain
        tmp = Dmt * torch.fft.fft(cztin)
        cztout = torch.fft.ifft(tmp)

        dataout = Bmt * cztout[:, 0:M]

        return dataout

    # @staticmethod
    def prechirpz(self, xsize, qsize, N, M):
        """
        Calculate the auxiliary vectors for chirp-z.

        Args:
            xsize (float): normalized pupil radius 1.0
            qsize (float): the original pixel number that could cover the region of interest
                in the image plane if using FFT
            N (int): the sampling number on the pupil
            M (int): the sampling number on the region of interest on the image plane

        Returns:
            (torch.Tensor,torch.Tensor,torch.Tensor): auxiliary vectors
        """

        L = N + M - 1
        sigma = 2 * np.pi * xsize * qsize / N / M
        Afac = torch.exp(2 * 1j * sigma * (1 - M))
        Bfac = torch.exp(2 * 1j * sigma * (1 - N))
        sqW = torch.exp(2 * 1j * sigma)
        W = sqW ** 2

        # fixed phase factor and amplitude factor
        Gfac = (2 * xsize / N) * torch.exp(1j * sigma * (1 - N) * (1 - M))

        # integration about n
        Utmp = torch.zeros([1, N], dtype=self.complex_type, device='cuda')
        A = torch.zeros([1, N], dtype=self.complex_type, device='cuda')
        Utmp[0, 0] = sqW * Afac
        A[0, 0] = 1.0
        for i in range(1, N):
            A[0, i] = Utmp[0, i - 1] * A[0, i - 1]
            Utmp[0, i] = Utmp[0, i - 1] * W

        #  the factor before the summation
        Utmp = torch.zeros([1, M], dtype=self.complex_type, device='cuda')
        B = torch.ones([1, M], dtype=self.complex_type, device='cuda')
        Utmp[0, 0] = sqW * Bfac
        B[0, 0] = Gfac
        for i in range(1, M):
            B[0, i] = Utmp[0, i - 1] * B[0, i - 1]
            Utmp[0, i] = Utmp[0, i - 1] * W

        # for circular convolution
        Utmp = torch.zeros([1, max(N, M) + 1], dtype=self.complex_type, device='cuda')
        Vtmp = torch.zeros([1, max(N, M) + 1], dtype=self.complex_type, device='cuda')
        Utmp[0, 0] = sqW
        Vtmp[0, 0] = 1.0
        # Utmp_cp = Utmp.clone()
        # Vtmp_cp = Vtmp.clone()
        for i in range(1, max(N, M) + 1):
            Vtmp[0, i] = Utmp[0, i - 1] * Vtmp[0, i - 1]
            Utmp[0, i] = Utmp[0, i - 1] * W
            # Vtmp[0, i] = Utmp_cp[0, i - 1] * Vtmp_cp[0, i - 1]
            # Utmp[0, i] = Utmp_cp[0, i - 1] * W
            # Vtmp_cp[0, i] = Vtmp[0, i].clone()
            # Utmp_cp[0, i] = Utmp[0, i].clone()

        D = torch.ones([1, L], dtype=self.complex_type, device='cuda')
        for i in range(0, M):
            D[0, i] = torch.conj(Vtmp[0, i])
        for i in range(0, N):
            D[0, L - 1 - i] = torch.conj(Vtmp[0, i + 1])

        D = torch.fft.fft(D, axis=1)

        return A, B, D

    # # old version
    # def _pre_compute(self):
    #     """
    #     Compute the common intermediate variables in advance, this can save time for PSFs simulation
    #     """
    #     # pupil radius (in diffraction units) and pupil coordinate sampling
    #     pupil_size = 1.0
    #     dxypupil = 2 * pupil_size / self.npupil
    #     xypupil = torch.arange(-pupil_size + dxypupil / 2, pupil_size, dxypupil, device='cuda', dtype=self.data_type)
    #     [xpupil, ypupil] = torch.meshgrid(xypupil, xypupil, indexing='ij')
    #     ypupil = torch.complex(ypupil, torch.zeros_like(ypupil))
    #     xpupil = torch.complex(xpupil, torch.zeros_like(xpupil))
    #
    #     # calculation of relevant Fresnel-coefficients for the interfaces
    #     costhetamed = torch.sqrt(1.0 - (xpupil ** 2 + ypupil ** 2) * (self.na ** 2) / (self.refmed ** 2))
    #     costhetacov = torch.sqrt(1.0 - (xpupil ** 2 + ypupil ** 2) * (self.na ** 2) / (self.refcov ** 2))
    #     costhetaimm = torch.sqrt(1.0 - (xpupil ** 2 + ypupil ** 2) * (self.na ** 2) / (self.refimm ** 2))
    #     fresnelpmedcov = 2 * self.refmed * costhetamed / (self.refmed * costhetacov + self.refcov * costhetamed)
    #     fresnelsmedcov = 2 * self.refmed * costhetamed / (self.refmed * costhetamed + self.refcov * costhetacov)
    #     fresnelpcovimm = 2 * self.refcov * costhetacov / (self.refcov * costhetaimm + self.refimm * costhetacov)
    #     fresnelscovimm = 2 * self.refcov * costhetacov / (self.refcov * costhetacov + self.refimm * costhetaimm)
    #     fresnelp = fresnelpmedcov * fresnelpcovimm
    #     fresnels = fresnelsmedcov * fresnelscovimm
    #
    #     # apodization
    #     apod = 1 / torch.sqrt(costhetaimm)
    #     # define aperture
    #     aperturemask = torch.where((xpupil ** 2 + ypupil ** 2).real < 1.0, 1.0, 0.0)
    #     self.amplitude = aperturemask * apod
    #
    #     # setting of vectorial functions
    #     phi = torch.atan2(torch.real(ypupil), torch.real(xpupil))
    #     cosphi = torch.cos(phi)
    #     sinphi = torch.sin(phi)
    #     costheta = costhetamed
    #     sintheta = torch.sqrt(1 - costheta ** 2)
    #
    #     pvec = torch.empty([3, self.npupil, self.npupil], dtype=self.complex_type, device='cuda')
    #     pvec[0] = fresnelp * costheta * cosphi
    #     pvec[1] = fresnelp * costheta * sinphi
    #     pvec[2] = -fresnelp * sintheta
    #     svec = torch.empty([3, self.npupil, self.npupil], dtype=self.complex_type, device='cuda')
    #     svec[0] = -fresnels * sinphi
    #     svec[1] = fresnels * cosphi
    #     svec[2] = 0 * cosphi
    #
    #     polarizationvector = torch.empty([2, 3, self.npupil, self.npupil], dtype=self.complex_type, device='cuda')
    #     for ipol in range(3):
    #         polarizationvector[0, ipol] = cosphi * pvec[ipol] - sinphi * svec[ipol]
    #         polarizationvector[1, ipol] = sinphi * pvec[ipol] + cosphi * svec[ipol]
    #
    #     self.wavevector = torch.empty([2, self.npupil, self.npupil], dtype=self.complex_type, device='cuda')
    #     self.wavevector[0] = 2 * np.pi * self.na / self.wavelength * xpupil
    #     self.wavevector[1] = 2 * np.pi * self.na / self.wavelength * ypupil
    #     self.wavevectorzimm = 2 * np.pi * self.refimm / self.wavelength * costhetaimm
    #     self.wavevectorzmed = 2 * np.pi * self.refmed / self.wavelength * costhetamed
    #
    #     # calculate aberration function
    #     waberration = torch.zeros_like(xpupil, dtype=self.complex_type, device='cuda')
    #     normfac = torch.sqrt(
    #         2 * (self.zernike_mode[:, 0] + 1) / (1 + torch.where(self.zernike_mode[:, 1] == 0, 1.0, 0.0)))
    #     zernikecoefs_norm = self.zernike_coef * normfac
    #     allzernikes = self.get_zernike(self.zernike_mode, xpupil, ypupil)
    #
    #     for izer in range(self.zernike_mode.shape[0]):
    #         waberration += zernikecoefs_norm[izer] * allzernikes[izer]
    #     waberration *= aperturemask
    #     self.zernike_phase = torch.exp(1j * 2 * np.pi * waberration / self.wavelength)
    #
    #     self.pupilmatrix = torch.empty([2, 3, self.npupil, self.npupil], dtype=self.complex_type, device='cuda')
    #     for imat in range(2):
    #         for jmat in range(3):
    #             self.pupilmatrix[imat, jmat] = self.amplitude * self.zernike_phase * polarizationvector[imat, jmat]
    #
    #     # czt transform(fft the pupil)
    #     xrange = self.pixel_size_xy[0] * self.psf_size / 2
    #     yrange = self.pixel_size_xy[1] * self.psf_size / 2
    #     imagesizex = xrange * self.na / self.wavelength
    #     imagesizey = yrange * self.na / self.wavelength
    #
    #     # calculate the auxiliary vectors for chirp-z, pixelsize_xy should be inverse to match row and column
    #     self.ax, self.bx, self.dx = self.prechirpz(pupil_size, imagesizey, self.npupil, self.psf_size)
    #     self.ay, self.by, self.dy = self.prechirpz(pupil_size, imagesizex, self.npupil, self.psf_size)
    #
    #     # calculate intensity normalization function using the PSF at focus
    #     fieldmatrix_norm = torch.empty([2, 3, self.psf_size, self.psf_size], dtype=self.complex_type, device='cuda')
    #     for itel in range(2):
    #         for jtel in range(3):
    #             Pupilfunction_norm = self.amplitude * polarizationvector[itel, jtel]
    #             inter_image_norm = torch.transpose(self.czt(Pupilfunction_norm, self.ax, self.bx, self.dx), 1, 0)
    #             fieldmatrix_norm[itel, jtel] = torch.transpose(self.czt(inter_image_norm, self.ay, self.by, self.dy), 1,
    #                                                            0)
    #     int_focus = torch.zeros([self.psf_size, self.psf_size], dtype=self.data_type, device='cuda')
    #     for jtel in range(3):
    #         for itel in range(2):
    #             int_focus += 1 / 3 * (torch.abs(fieldmatrix_norm[itel, jtel])) ** 2
    #     self.norm_intensity = torch.sum(int_focus)

    # parallel version
    def _pre_compute(self):
        """
        Compute the common intermediate variables in advance, this can save time for PSFs simulation
        """
        # pupil radius (in diffraction units) and pupil coordinate sampling
        pupil_size = 1.0
        dxypupil = 2 * pupil_size / self.npupil
        xypupil = torch.arange(-pupil_size + dxypupil / 2, pupil_size, dxypupil, device='cuda', dtype=self.data_type)
        [xpupil, ypupil] = torch.meshgrid(xypupil, xypupil, indexing='ij')
        ypupil = torch.complex(ypupil, torch.zeros_like(ypupil))
        xpupil = torch.complex(xpupil, torch.zeros_like(xpupil))

        # calculation of relevant Fresnel-coefficients for the interfaces
        costhetamed = torch.sqrt(1.0 - (xpupil ** 2 + ypupil ** 2) * (self.na ** 2) / (self.refmed ** 2))
        costhetacov = torch.sqrt(1.0 - (xpupil ** 2 + ypupil ** 2) * (self.na ** 2) / (self.refcov ** 2))
        costhetaimm = torch.sqrt(1.0 - (xpupil ** 2 + ypupil ** 2) * (self.na ** 2) / (self.refimm ** 2))
        fresnelpmedcov = 2 * self.refmed * costhetamed / (self.refmed * costhetacov + self.refcov * costhetamed)
        fresnelsmedcov = 2 * self.refmed * costhetamed / (self.refmed * costhetamed + self.refcov * costhetacov)
        fresnelpcovimm = 2 * self.refcov * costhetacov / (self.refcov * costhetaimm + self.refimm * costhetacov)
        fresnelscovimm = 2 * self.refcov * costhetacov / (self.refcov * costhetacov + self.refimm * costhetaimm)
        fresnelp = fresnelpmedcov * fresnelpcovimm
        fresnels = fresnelsmedcov * fresnelscovimm

        # apodization
        apod = 1 / torch.sqrt(costhetaimm)  # old fs version
        # apod = torch.sqrt(costhetaimm)/costhetamed  # sjoerd version
        # define aperture
        aperturemask = torch.where((xpupil ** 2 + ypupil ** 2).real < 1.0, 1.0, 0.0)
        self.amplitude = aperturemask * apod

        # setting of vectorial functions
        phi = torch.atan2(torch.real(ypupil), torch.real(xpupil))
        cosphi = torch.cos(phi)
        sinphi = torch.sin(phi)
        costheta = costhetamed
        sintheta = torch.sqrt(1 - costheta ** 2)

        pvec = torch.empty([3, self.npupil, self.npupil], dtype=self.complex_type, device='cuda')
        pvec[0] = fresnelp * costheta * cosphi
        pvec[1] = fresnelp * costheta * sinphi
        pvec[2] = -fresnelp * sintheta
        svec = torch.empty([3, self.npupil, self.npupil], dtype=self.complex_type, device='cuda')
        svec[0] = -fresnels * sinphi
        svec[1] = fresnels * cosphi
        svec[2] = 0 * cosphi

        polarizationvector = torch.empty([2, 3, self.npupil, self.npupil], dtype=self.complex_type, device='cuda')
        polarizationvector[0,] = cosphi * pvec - sinphi * svec
        polarizationvector[1,] = sinphi * pvec + cosphi * svec

        self.wavevector = torch.empty([2, self.npupil, self.npupil], dtype=self.complex_type, device='cuda')
        self.wavevector[0] = 2 * np.pi * self.na / self.wavelength * xpupil
        self.wavevector[1] = 2 * np.pi * self.na / self.wavelength * ypupil
        self.wavevectorzimm = 2 * np.pi * self.refimm / self.wavelength * costhetaimm
        self.wavevectorzmed = 2 * np.pi * self.refmed / self.wavelength * costhetamed

        # calculate aberration function
        waberration = torch.zeros_like(xpupil, dtype=self.complex_type, device='cuda')
        normfac = torch.sqrt(
            2 * (self.zernike_mode[:, 0] + 1) / (1 + torch.where(self.zernike_mode[:, 1] == 0, 1.0, 0.0)))
        zernikecoefs_norm = self.zernike_coef * normfac
        allzernikes = self.get_zernike(self.zernike_mode, xpupil, ypupil)

        waberration += torch.sum(zernikecoefs_norm[:, None, None]*allzernikes, dim=0)
        waberration *= aperturemask
        self.zernike_phase = torch.exp(1j * 2 * np.pi * waberration / self.wavelength)

        self.pupilmatrix = torch.empty([2, 3, self.npupil, self.npupil], dtype=self.complex_type, device='cuda')
        self.pupilmatrix = self.amplitude[None, None] * self.zernike_phase[None, None] * polarizationvector

        # czt transform(fft the pupil)
        xrange = self.pixel_size_xy[0] * self.psf_size / 2
        yrange = self.pixel_size_xy[1] * self.psf_size / 2
        imagesizex = xrange * self.na / self.wavelength
        imagesizey = yrange * self.na / self.wavelength

        # calculate the auxiliary vectors for chirp-z, pixelsize_xy should be inverse to match row and column
        self.ax, self.bx, self.dx = self.prechirpz(pupil_size, imagesizey, self.npupil, self.psf_size)
        self.ay, self.by, self.dy = self.prechirpz(pupil_size, imagesizex, self.npupil, self.psf_size)

        # calculate intensity normalization function using the PSF at focus
        fieldmatrix_norm = torch.empty([2, 3, self.psf_size, self.psf_size], dtype=self.complex_type, device='cuda')
        Pupilfunction_norm = self.amplitude[None, None, None] * polarizationvector[:, :, None]
        inter_image_norm = torch.transpose(self.czt_parallel(Pupilfunction_norm, self.ax, self.bx, self.dx), -1, -2)
        fieldmatrix_norm = torch.transpose(self.czt_parallel(inter_image_norm, self.ay, self.by, self.dy), -1, -2)

        int_focus = torch.zeros([self.psf_size, self.psf_size], dtype=self.data_type, device='cuda')
        int_focus += 1 / 3 * torch.sum(torch.abs(fieldmatrix_norm) ** 2, dim=(0, 1, 2))
        self.norm_intensity = torch.sum(int_focus)

    def simulate(self, x, y, z, photons, objstage=None):
        """
        Run the simulation to generate the vector PSFs with the given positions

        Args:
            x (torch.Tensor): x positions of the PSFs, unit nm
            y (torch.Tensor): y positions of the PSFs, unit nm
            z (torch.Tensor): z positions of the PSFs, unit nm
            photons (torch.Tensor): photon counts of the PSFs, unit photons
            objstage (torch.Tensor): objective stage positions relative to the cover-slip (0),
                the closer to the sample, the smaller this value is (-), unit nm

        Returns:
            torch.Tensor: PSFs, unit photons
        """

        n_mol = x.shape[0]
        objstage = torch.zeros(x.shape[0], dtype=self.data_type, device='cuda') if objstage is None else objstage

        field_matrix = torch.empty([2, 3, n_mol, self.psf_size, self.psf_size],
                                   dtype=self.complex_type, device='cuda')
        for jz in range(n_mol):
            # xyz induced phase, x,y should be inverse to match the column, row
            if z[jz] + self.zemit0 >= 0:
                phase_xyz = -y[jz] * self.wavevector[0] - x[jz] * self.wavevector[1] + \
                            (z[jz] + self.zemit0) * self.wavevectorzmed
                position_phase = torch.exp(1j * (phase_xyz + (objstage[jz] + self.objstage0) *
                                                 self.wavevectorzimm))
            else:
                # print("warning! the emitter's position may not have physical meaning")
                phase_xyz = -y[jz] * self.wavevector[0] - x[jz] * self.wavevector[1]
                position_phase = torch.exp(1j * (phase_xyz + (objstage[jz] + self.objstage0 + z[jz]
                                                              + self.zemit0) * self.wavevectorzimm))
            for itel in range(2):
                for jtel in range(3):
                    pupil_tmp = position_phase * self.pupilmatrix[itel, jtel]
                    inter_image = torch.transpose(self.czt(pupil_tmp, self.ay, self.by, self.dy), 1, 0)
                    field_matrix[itel, jtel, jz] = torch.transpose(self.czt(inter_image, self.ax, self.bx, self.dx), 1,
                                                                   0)

        psfs_out = torch.zeros([n_mol, self.psf_size, self.psf_size], device='cuda', dtype=torch.float32)
        for jz in range(n_mol):
            for jtel in range(3):
                for itel in range(2):
                    psfs_out[jz, :, :] += 1 / 3 * (torch.abs(field_matrix[itel, jtel, jz])) ** 2

        psfs_out /= self.norm_intensity

        # otf rescale
        if self.otf_rescale_xy[0] or self.otf_rescale_xy[1] != 0:
            psfs_out = self.otf_rescale(psfdata=psfs_out, sigma_xy=self.otf_rescale_xy)

        # normalize the psf to 1, then multiply with the photon number
        # psfs_out /= psfs_out.sum(-1).sum(-1)[:, None, None]
        psfs_out *= photons[:, None, None]

        return psfs_out

    def czt_parallel(self, datain, A, B, D):
        """
        Execute the chirp-z transform

        Args:
            datain (torch.Tensor): input data, 2D array
            A (torch.Tensor): chirp parameter, 1D array
            B (torch.Tensor): chirp parameter, 1D array
            D (torch.Tensor): chirp parameter, 1D array

        Returns:
            torch.Tensor: output data, 2D array
        """

        N = A.shape[1]
        M = B.shape[1]
        L = D.shape[1]
        K = datain.shape[-2]
        n_mol = datain.shape[-3]

        # torch.repeat_interleave is too slow
        # t0 = time.time()
        # Amt = torch.repeat_interleave(A, K, 0)
        # Bmt = torch.repeat_interleave(B, K, 0)
        # Dmt = torch.repeat_interleave(D, K, 0)
        # print('torch czt: ', time.time() - t0)
        Amt = A.expand(K, N)
        Bmt = B.expand(K, M)
        Dmt = D.expand(K, L)

        cztin = torch.zeros([2, 3, n_mol, K, L], dtype=self.complex_type, device='cuda')
        cztin[:, :, :, :, 0:N] = Amt[None, None, None] * datain
        try:
            tmp = Dmt * torch.fft.fft(cztin, dim=-1)
        except:
            print('fft error')
        cztout = torch.fft.ifft(tmp, dim=-1)

        dataout = Bmt[None, None, None] * cztout[:, :, :, :, 0:M]

        return dataout

    def simulate_parallel(self, x, y, z, photons, objstage=None):
        """
        Run the simulation to generate the vector PSFs with the given positions

        Args:
            x (torch.Tensor): x positions of the PSFs, unit nm
            y (torch.Tensor): y positions of the PSFs, unit nm
            z (torch.Tensor): z positions of the PSFs, unit nm
            photons (torch.Tensor): photon counts of the PSFs, unit photons
            objstage (torch.Tensor): objective stage positions relative to the cover-slip (0),
                the closer to the sample, the smaller this value is (-), unit nm

        Returns:
            torch.Tensor: PSFs, unit photons
        """

        n_mol = x.shape[0]
        if n_mol == 0:
            return torch.zeros([0, self.psf_size, self.psf_size], device='cuda', dtype=torch.float32)

        objstage = torch.zeros(x.shape[0], dtype=self.data_type, device='cuda') if objstage is None else objstage

        # batch_size in parallel to save GPU memory
        slice_list = []
        batch_size = 100
        for i in np.arange(0, n_mol, batch_size):
            slice_list.append(slice(i, min(i + batch_size, n_mol)))

        psfs_out = torch.zeros([n_mol, self.psf_size, self.psf_size], device='cuda', dtype=torch.float32)
        for slice_tmp in slice_list:
            length_tmp = slice_tmp.stop - slice_tmp.start
            position_phase = torch.empty([length_tmp, self.npupil, self.npupil], dtype=self.complex_type, device='cuda')

            idx = torch.where(z[slice_tmp] + self.zemit0 >= 0)[0]
            phase_xyz_tmp = -y[slice_tmp][idx][:, None, None] * self.wavevector[0][None] - x[slice_tmp][idx][:, None, None] * \
                            self.wavevector[1][None] + \
                            (z[slice_tmp][idx] + self.zemit0)[:, None, None] * self.wavevectorzmed[None]
            position_phase[idx, :, :] = torch.exp(
                1j * (phase_xyz_tmp + (objstage[slice_tmp][idx][:, None, None] + self.objstage0) *
                      self.wavevectorzimm[None]))

            idx = torch.where(z[slice_tmp] + self.zemit0 < 0)[0]
            phase_xyz_tmp = -y[slice_tmp][idx][:, None, None] * self.wavevector[0][None] - x[slice_tmp][idx][:, None, None] * \
                            self.wavevector[1][None]
            position_phase[idx, :, :] = torch.exp(
                1j * (phase_xyz_tmp + (objstage[slice_tmp][idx][:, None, None] + self.objstage0 + z[slice_tmp][idx][:, None, None]
                                       + self.zemit0) * self.wavevectorzimm[None]))

            pupil_tmp = position_phase[None, None] * self.pupilmatrix[:, :, None]
            inter_image = torch.transpose(self.czt_parallel(pupil_tmp, self.ay, self.by, self.dy), -1, -2)
            field_matrix = torch.transpose(self.czt_parallel(inter_image, self.ax, self.bx, self.dx), -1, -2)

            psfs_out[slice_tmp] += 1 / 3 * torch.sum((torch.abs(field_matrix[:, :])) ** 2, dim=(0, 1))

        # intensity normalization
        psfs_out /= self.norm_intensity

        # otf rescale
        if self.otf_rescale_xy[0] or self.otf_rescale_xy[1] != 0:
            psfs_out = self.otf_rescale(psfdata=psfs_out, sigma_xy=self.otf_rescale_xy)

        # normalize the psf to 1, then multiply with the photon number
        psfs_out /= psfs_out.sum(-1).sum(-1)[:, None, None]
        psfs_out *= photons[:, None, None]

        return psfs_out

    def compute_crlb(self, x, y, z, photons, bgs):
        """
        Calculate the CRLB of this PSF model at give positions, photons and backgrounds.

        Args:
            x (torch.Tensor): x positions of the PSFs, unit nm
            y (torch.Tensor): y positions of the PSFs, unit nm
            z (torch.Tensor): z positions of the PSFs, unit nm
            photons (torch.Tensor): photon counts of the PSFs, unit photons
            bgs (torch.Tensor): background counts of the PSFs, unit photons

        Returns:
            (torch.Tensor,torch.Tensor): CRLB xyz (nPSFs, 3) and model PSFs, unit nm and photons
        """

        n_mol = x.shape[0]
        # TODO: global fitting based on shared control flag, not implemented yet
        # num_zernike = self.zernike_mode.shape[0]
        # # shared flag about each parameter type, 1 for shared and 0 for not shared
        # shared_flag = np.concatenate((np.ones(num_zernike), np.array([0, 0, 0, 0, 0])), axis=0)
        # num_pars = int((num_zernike + 5) * n_mol - shared_flag.sum() * (n_mol - 1))
        # # parameter map: [shared, parameter type, relevant channel]
        # param_map = np.zeros((num_pars, 3), dtype=int)
        # n = 0
        # for i in range(len(shared_flag)):
        #     if shared_flag[i] == 1:
        #         # first column represents this parameter is shared or not
        #         param_map[n, 0] = 1
        #         # second column represents the type of the parameter, 1:num_zernike for zernike,
        #         # num_zernike+1:num_zernike+5 for x,y,z,photons,background
        #         param_map[n, 1] = i + 1
        #         # third column represents the index of the data related to this parameter,
        #         # 0 means all data are relevant
        #         param_map[n, 2] = 0
        #         n += 1
        #     elif shared_flag[i] == 0:
        #         # if not shared, means there are n_mol parameters of this type
        #         for j in range(n_mol):
        #             # the same as above
        #             param_map[n, 0] = 0
        #             param_map[n, 1] = i + 1
        #             param_map[n, 2] = j + 1
        #             n += 1

        # calculate the derivatives
        [dudt, model] = self._compute_derivative(x, y, z, photons, bgs)
        # dudt = self._torch_jacobian_derivative(x, y, z, photons, bgs)

        # calculate hessian matrix, here only consider not shared parameters: x, y, z, photons, background
        num_pars = n_mol * 5
        t2 = 1 / model
        hessian = torch.zeros([num_pars, num_pars], device='cuda', dtype=self.data_type)
        for p1 in range(num_pars):
            temp1_zind = int(np.floor(p1 / 5))  # the index of data
            temp1_pind = int(p1 % 5)  # the index of parameter type
            temp1 = dudt[:, :, :, temp1_pind]  # the derivative of data concerning this parameter type
            for p2 in range(p1, num_pars):
                temp2_zind = int(np.floor(p2 / 5))
                temp2_pind = int(p2 % 5)
                temp2 = dudt[:, :, :, temp2_pind]

                # since all parameters are not shared, only the same molecule data makes sense
                # when multiply gradients of two parameters
                if temp1_zind == temp2_zind:
                    temp = t2[temp1_zind, :, :] * temp1[temp1_zind, :, :] * temp2[temp2_zind, :, :]
                    hessian[p1, p2] = torch.sum(temp)
                    hessian[p2, p1] = hessian[p1, p2]

        # calculate local fisher matrix and crlb
        xyz_crlb = torch.zeros([n_mol, 3], device='cuda')
        for j in range(n_mol):
            fisher_tmp = hessian[j * 5:j * 5 + 5, j * 5:j * 5 + 5]
            sqrt_crlb_tmp = torch.sqrt(torch.diag(torch.inverse(fisher_tmp)))
            xyz_crlb[j] = sqrt_crlb_tmp[0:3]

        return xyz_crlb, model

    def _compute_derivative(self, x, y, z, photons, bgs):
        # todo: compute in parallel, refer to simulate_parallel
        """Calculate the analytical derivatives of the PSFs at given parameters with respect to x,y,z,photons,bg"""
        n_mol = x.shape[0]
        objstage = torch.zeros(x.shape[0], dtype=self.data_type, device='cuda')
        field_matrix = torch.empty([2, 3, n_mol, self.psf_size, self.psf_size],
                                   dtype=self.complex_type, device='cuda')
        field_matrix_ders = torch.empty([2, 3, n_mol, 3, self.psf_size, self.psf_size],
                                        dtype=self.complex_type, device='cuda')
        for jz in range(n_mol):
            # xyz induced phase
            if z[jz] + self.zemit0 >= 0:
                phase_xyz = -y[jz] * self.wavevector[0] - x[jz] * self.wavevector[1] + \
                            (z[jz] + self.zemit0) * self.wavevectorzmed
                position_phase = torch.exp(1j * (phase_xyz + (objstage[jz] + self.objstage0) *
                                                 self.wavevectorzimm))
            else:
                # print("warning! the emitter's position may not have physical meaning")
                phase_xyz = -y[jz] * self.wavevector[0] - x[jz] * self.wavevector[1]
                position_phase = torch.exp(1j * (phase_xyz + (objstage[jz] + self.objstage0 + z[jz]
                                                              + self.zemit0) * self.wavevectorzimm))
            for itel in range(2):
                for jtel in range(3):
                    pupil_tmp = position_phase * self.pupilmatrix[itel, jtel]
                    inter_image = torch.transpose(self.czt(pupil_tmp, self.ay, self.by, self.dy), 1, 0)
                    field_matrix[itel, jtel, jz] = torch.transpose(self.czt(inter_image, self.ax, self.bx, self.dx), 1,
                                                                   0)
                    # derivatives with respect to x,y,z
                    pupilfunction_x = -1j * self.wavevector[1] * position_phase * self.pupilmatrix[itel, jtel]
                    inter_image_x = torch.transpose(self.czt(pupilfunction_x, self.ay, self.by, self.dy), 1, 0)
                    field_matrix_ders[itel, jtel, jz, 0] = torch.transpose(self.czt(inter_image_x, self.ax,
                                                                                    self.bx, self.dx), 1, 0)

                    pupilfunction_y = -1j * self.wavevector[0] * position_phase * self.pupilmatrix[itel, jtel]
                    inter_image_y = torch.transpose(self.czt(pupilfunction_y, self.ay, self.by, self.dy), 1, 0)
                    field_matrix_ders[itel, jtel, jz, 1] = torch.transpose(self.czt(inter_image_y, self.ax,
                                                                                    self.bx, self.dx), 1, 0)

                    if z[jz] + self.zemit0 >= 0:
                        pupilfunction_z = 1j * self.wavevectorzmed * position_phase * self.pupilmatrix[itel, jtel]
                    else:
                        pupilfunction_z = 1j * self.wavevectorzimm * position_phase * self.pupilmatrix[itel, jtel]
                    inter_image_z = torch.transpose(self.czt(pupilfunction_z, self.ay, self.by, self.dy), 1, 0)
                    field_matrix_ders[itel, jtel, jz, 2] = torch.transpose(self.czt(inter_image_z, self.ax,
                                                                                    self.bx, self.dx), 1, 0)

        psfs = torch.zeros([n_mol, self.psf_size, self.psf_size], device='cuda', dtype=self.data_type)
        psfs_ders = torch.zeros([n_mol, self.psf_size, self.psf_size, 3], device='cuda', dtype=self.data_type)
        for jz in range(n_mol):
            for jtel in range(3):
                for itel in range(2):
                    psfs[jz, :, :] += 1 / 3 * (torch.abs(field_matrix[itel, jtel, jz])) ** 2
                    for jder in range(3):
                        psfs_ders[jz, :, :, jder] = psfs_ders[jz, :, :, jder] + 2 / 3 * \
                                                    torch.real(torch.conj(field_matrix[itel, jtel, jz]) *
                                                               field_matrix_ders[itel, jtel, jz, jder])
        psfs /= self.norm_intensity
        psfs_ders /= self.norm_intensity

        # otf rescale
        if self.otf_rescale_xy[0] or self.otf_rescale_xy[1] != 0:
            psfs = self.otf_rescale(psfdata=psfs, sigma_xy=self.otf_rescale_xy)
            psfs_ders[:, :, :, 0] = self.otf_rescale(psfdata=psfs_ders[:, :, :, 0], sigma_xy=self.otf_rescale_xy)
            psfs_ders[:, :, :, 1] = self.otf_rescale(psfdata=psfs_ders[:, :, :, 1], sigma_xy=self.otf_rescale_xy)
            psfs_ders[:, :, :, 2] = self.otf_rescale(psfdata=psfs_ders[:, :, :, 2], sigma_xy=self.otf_rescale_xy)

        psfs_out = gpu(psfs * photons[:, None, None] + bgs[:, None, None])
        ders_out = torch.zeros([n_mol, self.psf_size, self.psf_size, 5], device='cuda', dtype=self.data_type)
        ders_out[:, :, :, 0:3] = psfs_ders * photons[:, None, None, None]
        ders_out[:, :, :, 3] = psfs
        ders_out[:, :, :, 4] = torch.ones_like(psfs)
        return ders_out, psfs_out

    def _torch_jacobian_derivative(self, x, y, z, photons, bgs):
        """Calculate the derivatives of the PSFs at given parameters with respect to x,y,z,photons,bg
        using torch.autograd.functional.jacobian"""
        x.requires_grad = True
        y.requires_grad = True
        z.requires_grad = True
        photons.requires_grad = True
        bgs.requires_grad = True
        jacobian = torch.autograd.functional.jacobian(self.simulate, (x, y, z, photons))

        ders_out = torch.zeros([x.shape[0], self.psf_size, self.psf_size, 5], device='cuda', dtype=self.data_type)
        for i in range(len(jacobian)):
            for j in range(x.shape[0]):
                ders_out[j, :, :, i] = jacobian[i][j, :, :, j]
        ders_out[:, :, :, 4] = torch.ones([x.shape[0], self.psf_size, self.psf_size])

        return ders_out

    def optimize_crlb(self, x, y, z, photons, bgs, tolerance):
        """
        Optimize the CRLB of this PSF model with respect to zernike coefficients
        at give positions, photons and backgrounds. The instance should be initialized with req_grad=True

        Args:
            x (torch.Tensor): x positions of the PSFs, unit nm
            y (torch.Tensor): y positions of the PSFs, unit nm
            z (torch.Tensor): z positions of the PSFs, unit nm
            photons (torch.Tensor): photon counts of the PSFs, unit photon
            bgs (torch.Tensor): background counts of the PSFs, unit photon
            tolerance (float): stop criteria, the difference of CRLB between two iterations

        Returns:

        """

        n_mol = x.shape[0]
        crlb_optimizer = torch.optim.Adam([self.zernike_coef], lr=0.1)
        loss_1 = 1e6
        iter_num = 1
        # for iter_num in range(iterations):
        while True:
            xyz_crlb, model = self.compute_crlb(x, y, z, photons, bgs)
            crlb_3d_avg = torch.sum(xyz_crlb[:, 0]**2 + xyz_crlb[:, 1]**2 + xyz_crlb[:, 2]**2)/n_mol
            print(f"iter: {iter_num}, crlb_3d_avg: {crlb_3d_avg}")
            crlb_optimizer.zero_grad()
            crlb_3d_avg.backward()
            crlb_optimizer.step()
            self._pre_compute()
            if torch.abs((loss_1 - crlb_3d_avg) / loss_1) < tolerance:
                break
            loss_1 = crlb_3d_avg.detach()
            iter_num += 1
        print('CRLB optimization done')



class VectorPSFTorchFit(VectorPSF):
    """
    Vector psf simulator using pytorch, thus psf parameters can be optimized by pytorch
    """

    def __init__(self, psf_params, req_grad=False, data_type=torch.float64):
        """

        Args:
            psf_params (dict): PSF parameters for simulation except emitter positions
            req_grad (bool): whether the PSF parameters are required gradient
            data_type (torch.dtype): data type for the PSF parameters

        Returns:
            VectorPSFTorch: an instance of VectorPSFTorch
        """

        self.data_type = data_type
        if data_type == torch.float64:
            self.complex_type = torch.complex128
        elif data_type == torch.float32:
            self.complex_type = torch.complex64
        elif data_type == torch.float16:
            self.complex_type = torch.complex32
        else:
            raise ValueError(f'unsupported data type {data_type}')

        self.na = torch.tensor(psf_params['na'], device='cuda', dtype=self.data_type)
        self.wavelength = torch.tensor(psf_params['wavelength'], device='cuda', dtype=self.data_type)
        self.refmed = torch.tensor(psf_params['refmed'], device='cuda', dtype=self.data_type,)
        self.refcov = torch.tensor(psf_params['refcov'], device='cuda', dtype=self.data_type,)
        self.refimm = torch.tensor(psf_params['refimm'], device='cuda', dtype=self.data_type,)
        self.zernike_mode = torch.tensor(psf_params['zernike_mode'], device='cuda', dtype=self.data_type)
        self.zernike_coef = torch.tensor(psf_params['zernike_coef'], device='cuda', dtype=self.data_type,
                                         requires_grad=req_grad)
        self.zernike_coef_map = None
        self.objstage0 = torch.tensor(psf_params['objstage0'], device='cuda', dtype=self.data_type,)
        try:
            self.zemit0 = torch.tensor(psf_params['zemit0'], device='cuda', dtype=self.data_type, )
        except KeyError:
            self.zemit0 = torch.tensor(-psf_params['objstage0']/psf_params['refimm']*psf_params['refmed'],
                                       device='cuda',
                                       dtype=self.data_type,)

        self.pixel_size_xy = torch.tensor(psf_params['pixel_size_xy'], device='cuda', dtype=self.data_type)
        self.otf_rescale_xy = torch.tensor(psf_params['otf_rescale_xy'], device='cuda', dtype=self.data_type,)
        self.npupil = psf_params['npupil']
        self.psf_size = psf_params['psf_size']

        self._pre_compute()

    @staticmethod
    def get_zernike(orders, xpupil, ypupil):
        """
        Calculate zernike polynomials on pupil plane

        Args:
            orders (torch.Tensor): zernike orders, 2D array, first column is radial order,
                second column is azimuthal order
            xpupil (torch.Tensor): x coordinate of pupil plane
            ypupil (torch.Tensor): y coordinate of pupil plane

        Returns:
            torch.Tensor: zernike polynomials
        """

        xpupil = torch.real(xpupil)
        ypupil = torch.real(ypupil)
        zersize = orders.shape
        Nzer = zersize[0]
        radormax = int(max(orders[:, 0]))
        azormax = int(max(abs(orders[:, 1])))
        [Nx, Ny] = xpupil.shape

        # zerpol = np.zeros( [radormax+1,azormax+1,Nx,Ny] )
        zerpol = torch.zeros([21, 6, Nx, Ny], device='cuda')
        rhosq = xpupil ** 2 + ypupil ** 2
        rho = torch.sqrt(rhosq)
        zerpol[0, 0, :, :] = torch.ones_like(xpupil)

        for jm in range(1, azormax + 2 + 1):
            m = jm - 1
            if m > 0:
                zerpol[jm - 1, jm - 1, :, :] = rho * torch.squeeze(zerpol[jm - 1 - 1, jm - 1 - 1, :, :])

            zerpol[jm + 2 - 1, jm - 1, :, :] = ((m + 2) * rhosq - m - 1) * torch.squeeze(zerpol[jm - 1, jm - 1, :, :])
            for p in range(2, radormax - m + 2 + 1):
                n = m + 2 * p
                jn = n + 1
                zerpol[jn - 1, jm - 1, :, :] = (2 * (n - 1) * (n * (n - 2) * (2 * rhosq - 1) - m ** 2) * torch.squeeze(
                    zerpol[jn - 2 - 1, jm - 1, :, :]) -
                                                n * (n + m - 2) * (n - m - 2) * torch.squeeze(
                            zerpol[jn - 4 - 1, jm - 1, :, :])) / ((n - 2) * (n + m) * (n - m))

        phi = torch.atan2(ypupil, xpupil)
        allzernikes = torch.zeros([Nzer, Nx, Ny], device='cuda')
        for j in range(1, Nzer + 1):
            n = int(orders[j - 1, 0])
            m = int(orders[j - 1, 1])
            if m >= 0:
                allzernikes[j - 1, :, :] = torch.squeeze(zerpol[n + 1 - 1, m + 1 - 1, :, :]) * torch.cos(m * phi)
            else:
                allzernikes[j - 1, :, :] = torch.squeeze(zerpol[n + 1 - 1, -m + 1 - 1, :, :]) * torch.sin(-m * phi)

        # plt.figure(constrained_layout=True)
        # for i in range(21):
        #     plt.subplot(3, 7, i + 1)
        #     plt.imshow(cpu(allzernikes[i]))
        # plt.show()

        return allzernikes

    def czt(self, datain, A, B, D):
        """
        Execute the chirp-z transform

        Args:
            datain (torch.Tensor): input data, 2D array
            A (torch.Tensor): chirp parameter, 1D array
            B (torch.Tensor): chirp parameter, 1D array
            D (torch.Tensor): chirp parameter, 1D array

        Returns:
            torch.Tensor: output data, 2D array
        """

        N = A.shape[1]
        M = B.shape[1]
        L = D.shape[1]
        K = datain.shape[0]

        # torch.repeat_interleave is too slow
        # t0 = time.time()
        # Amt = torch.repeat_interleave(A, K, 0)
        # Bmt = torch.repeat_interleave(B, K, 0)
        # Dmt = torch.repeat_interleave(D, K, 0)
        # print('torch czt: ', time.time() - t0)
        Amt = A.expand(K, N)
        Bmt = B.expand(K, M)
        Dmt = D.expand(K, L)

        cztin = torch.zeros([K, L], dtype=self.complex_type, device='cuda')
        cztin[:, 0:N] = Amt * datain
        tmp = Dmt * torch.fft.fft(cztin)
        cztout = torch.fft.ifft(tmp)

        dataout = Bmt * cztout[:, 0:M]

        return dataout

    def czt_parallel(self, datain, A, B, D):
        """
        Execute the chirp-z transform

        Args:
            datain (torch.Tensor): input data, 2D array
            A (torch.Tensor): chirp parameter, 1D array
            B (torch.Tensor): chirp parameter, 1D array
            D (torch.Tensor): chirp parameter, 1D array

        Returns:
            torch.Tensor: output data, 2D array
        """

        N = A.shape[1]
        M = B.shape[1]
        L = D.shape[1]
        K = datain.shape[-2]
        n_mol = datain.shape[-3]

        # torch.repeat_interleave is too slow
        # t0 = time.time()
        # Amt = torch.repeat_interleave(A, K, 0)
        # Bmt = torch.repeat_interleave(B, K, 0)
        # Dmt = torch.repeat_interleave(D, K, 0)
        # print('torch czt: ', time.time() - t0)
        Amt = A.expand(K, N)
        Bmt = B.expand(K, M)
        Dmt = D.expand(K, L)

        cztin = torch.zeros([2, 3, n_mol, K, L], dtype=self.complex_type, device='cuda')
        cztin[:, :, :, :, 0:N] = Amt[None, None, None] * datain
        try:
            tmp = Dmt * torch.fft.fft(cztin, dim=-1)
        except:
            print('fft error')
        cztout = torch.fft.ifft(tmp, dim=-1)

        dataout = Bmt[None, None, None] * cztout[:, :, :, :, 0:M]

        return dataout

    def prechirpz(self, xsize, qsize, N, M):
        """
        Calculate the auxiliary vectors for chirp-z.

        Args:
            xsize (float): normalized pupil radius 1.0
            qsize (float): the original pixel number that could cover the region of interest
                in the image plane if using FFT
            N (int): the sampling number on the pupil
            M (int): the sampling number on the region of interest on the image plane

        Returns:
            (torch.Tensor,torch.Tensor,torch.Tensor): auxiliary vectors
        """

        L = N + M - 1
        sigma = 2 * np.pi * xsize * qsize / N / M
        Afac = torch.exp(2 * 1j * sigma * (1 - M))
        Bfac = torch.exp(2 * 1j * sigma * (1 - N))
        sqW = torch.exp(2 * 1j * sigma)
        W = sqW ** 2

        # fixed phase factor and amplitude factor
        Gfac = (2 * xsize / N) * torch.exp(1j * sigma * (1 - N) * (1 - M))

        # integration about n
        Utmp = torch.zeros([1, N], dtype=self.complex_type, device='cuda')
        A = torch.zeros([1, N], dtype=self.complex_type, device='cuda')
        Utmp[0, 0] = sqW * Afac
        A[0, 0] = 1.0
        for i in range(1, N):
            A[0, i] = Utmp[0, i - 1] * A[0, i - 1]
            Utmp[0, i] = Utmp[0, i - 1] * W

        #  the factor before the summation
        Utmp = torch.zeros([1, M], dtype=self.complex_type, device='cuda')
        B = torch.ones([1, M], dtype=self.complex_type, device='cuda')
        Utmp[0, 0] = sqW * Bfac
        B[0, 0] = Gfac
        for i in range(1, M):
            B[0, i] = Utmp[0, i - 1] * B[0, i - 1]
            Utmp[0, i] = Utmp[0, i - 1] * W

        # for circular convolution
        Utmp = torch.zeros([1, max(N, M) + 1], dtype=self.complex_type, device='cuda')
        Vtmp = torch.zeros([1, max(N, M) + 1], dtype=self.complex_type, device='cuda')
        Utmp[0, 0] = sqW
        Vtmp[0, 0] = 1.0
        # Utmp_cp = Utmp.clone()
        # Vtmp_cp = Vtmp.clone()
        for i in range(1, max(N, M) + 1):
            Vtmp[0, i] = Utmp[0, i - 1] * Vtmp[0, i - 1]
            Utmp[0, i] = Utmp[0, i - 1] * W
            # Vtmp[0, i] = Utmp_cp[0, i - 1] * Vtmp_cp[0, i - 1]
            # Utmp[0, i] = Utmp_cp[0, i - 1] * W
            # Vtmp_cp[0, i] = Vtmp[0, i].clone()
            # Utmp_cp[0, i] = Utmp[0, i].clone()

        D = torch.ones([1, L], dtype=self.complex_type, device='cuda')
        for i in range(0, M):
            D[0, i] = torch.conj(Vtmp[0, i])
        for i in range(0, N):
            D[0, L - 1 - i] = torch.conj(Vtmp[0, i + 1])

        D = torch.fft.fft(D, axis=1)

        return A, B, D

    # old version
    @deprecated(reason="the same as matlab code, using for loop is slow")
    def _pre_compute_v1(self):
        """
        Compute the common intermediate variables in advance, this can save time for PSFs simulation
        """
        # pupil radius (in diffraction units) and pupil coordinate sampling
        pupil_size = 1.0
        dxypupil = 2 * pupil_size / self.npupil
        xypupil = torch.arange(-pupil_size + dxypupil / 2, pupil_size, dxypupil, device='cuda', dtype=self.data_type)
        [xpupil, ypupil] = torch.meshgrid(xypupil, xypupil, indexing='ij')
        ypupil = torch.complex(ypupil, torch.zeros_like(ypupil))
        xpupil = torch.complex(xpupil, torch.zeros_like(xpupil))

        # calculation of relevant Fresnel-coefficients for the interfaces
        costhetamed = torch.sqrt(1.0 - (xpupil ** 2 + ypupil ** 2) * (self.na ** 2) / (self.refmed ** 2))
        costhetacov = torch.sqrt(1.0 - (xpupil ** 2 + ypupil ** 2) * (self.na ** 2) / (self.refcov ** 2))
        costhetaimm = torch.sqrt(1.0 - (xpupil ** 2 + ypupil ** 2) * (self.na ** 2) / (self.refimm ** 2))
        fresnelpmedcov = 2 * self.refmed * costhetamed / (self.refmed * costhetacov + self.refcov * costhetamed)
        fresnelsmedcov = 2 * self.refmed * costhetamed / (self.refmed * costhetamed + self.refcov * costhetacov)
        fresnelpcovimm = 2 * self.refcov * costhetacov / (self.refcov * costhetaimm + self.refimm * costhetacov)
        fresnelscovimm = 2 * self.refcov * costhetacov / (self.refcov * costhetacov + self.refimm * costhetaimm)
        fresnelp = fresnelpmedcov * fresnelpcovimm
        fresnels = fresnelsmedcov * fresnelscovimm

        # apodization
        apod = 1 / torch.sqrt(costhetaimm)
        # define aperture
        aperturemask = torch.where((xpupil ** 2 + ypupil ** 2).real < 1.0, 1.0, 0.0)
        self.amplitude = aperturemask * apod

        # setting of vectorial functions
        phi = torch.atan2(torch.real(ypupil), torch.real(xpupil))
        cosphi = torch.cos(phi)
        sinphi = torch.sin(phi)
        costheta = costhetamed
        sintheta = torch.sqrt(1 - costheta ** 2)

        pvec = torch.empty([3, self.npupil, self.npupil], dtype=self.complex_type, device='cuda')
        pvec[0] = fresnelp * costheta * cosphi
        pvec[1] = fresnelp * costheta * sinphi
        pvec[2] = -fresnelp * sintheta
        svec = torch.empty([3, self.npupil, self.npupil], dtype=self.complex_type, device='cuda')
        svec[0] = -fresnels * sinphi
        svec[1] = fresnels * cosphi
        svec[2] = 0 * cosphi

        polarizationvector = torch.empty([2, 3, self.npupil, self.npupil], dtype=self.complex_type, device='cuda')
        for ipol in range(3):
            polarizationvector[0, ipol] = cosphi * pvec[ipol] - sinphi * svec[ipol]
            polarizationvector[1, ipol] = sinphi * pvec[ipol] + cosphi * svec[ipol]

        self.wavevector = torch.empty([2, self.npupil, self.npupil], dtype=self.complex_type, device='cuda')
        self.wavevector[0] = 2 * np.pi * self.na / self.wavelength * xpupil
        self.wavevector[1] = 2 * np.pi * self.na / self.wavelength * ypupil
        self.wavevectorzimm = 2 * np.pi * self.refimm / self.wavelength * costhetaimm
        self.wavevectorzmed = 2 * np.pi * self.refmed / self.wavelength * costhetamed

        # calculate aberration function
        waberration = torch.zeros_like(xpupil, dtype=self.complex_type, device='cuda')
        normfac = torch.sqrt(
            2 * (self.zernike_mode[:, 0] + 1) / (1 + torch.where(self.zernike_mode[:, 1] == 0, 1.0, 0.0)))
        zernikecoefs_norm = self.zernike_coef * normfac
        allzernikes = self.get_zernike(self.zernike_mode, xpupil, ypupil)

        for izer in range(self.zernike_mode.shape[0]):
            waberration += zernikecoefs_norm[izer] * allzernikes[izer]
        waberration *= aperturemask
        self.zernike_phase = torch.exp(1j * 2 * np.pi * waberration / self.wavelength)

        self.pupilmatrix = torch.empty([2, 3, self.npupil, self.npupil], dtype=self.complex_type, device='cuda')
        for imat in range(2):
            for jmat in range(3):
                self.pupilmatrix[imat, jmat] = self.amplitude * self.zernike_phase * polarizationvector[imat, jmat]

        # czt transform(fft the pupil)
        xrange = self.pixel_size_xy[0] * self.psf_size / 2
        yrange = self.pixel_size_xy[1] * self.psf_size / 2
        imagesizex = xrange * self.na / self.wavelength
        imagesizey = yrange * self.na / self.wavelength

        # calculate the auxiliary vectors for chirp-z, pixelsize_xy should be inverse to match row and column
        self.ax, self.bx, self.dx = self.prechirpz(pupil_size, imagesizey, self.npupil, self.psf_size)
        self.ay, self.by, self.dy = self.prechirpz(pupil_size, imagesizex, self.npupil, self.psf_size)

        # calculate intensity normalization function using the PSF at focus
        fieldmatrix_norm = torch.empty([2, 3, self.psf_size, self.psf_size], dtype=self.complex_type, device='cuda')
        for itel in range(2):
            for jtel in range(3):
                Pupilfunction_norm = self.amplitude * polarizationvector[itel, jtel]
                inter_image_norm = torch.transpose(self.czt(Pupilfunction_norm, self.ax, self.bx, self.dx), 1, 0)
                fieldmatrix_norm[itel, jtel] = torch.transpose(self.czt(inter_image_norm, self.ay, self.by, self.dy), 1,
                                                               0)
        int_focus = torch.zeros([self.psf_size, self.psf_size], dtype=self.data_type, device='cuda')
        for jtel in range(3):
            for itel in range(2):
                int_focus += 1 / 3 * (torch.abs(fieldmatrix_norm[itel, jtel])) ** 2
        self.norm_intensity = torch.sum(int_focus)

    @deprecated(reason="parallel version of v1, but not compatible with simulate v3, where zernike phase "
                       "is not computed in advance")
    def _pre_compute_v2(self):
        """
        Compute the common intermediate variables in advance, this can save time for PSFs simulation
        """
        # pupil radius (in diffraction units) and pupil coordinate sampling
        pupil_size = 1.0
        dxypupil = 2 * pupil_size / self.npupil
        xypupil = torch.arange(-pupil_size + dxypupil / 2, pupil_size, dxypupil, device='cuda', dtype=self.data_type)
        [xpupil, ypupil] = torch.meshgrid(xypupil, xypupil, indexing='ij')
        ypupil = torch.complex(ypupil, torch.zeros_like(ypupil))
        xpupil = torch.complex(xpupil, torch.zeros_like(xpupil))

        # calculation of relevant Fresnel-coefficients for the interfaces
        costhetamed = torch.sqrt(1.0 - (xpupil ** 2 + ypupil ** 2) * (self.na ** 2) / (self.refmed ** 2))
        costhetacov = torch.sqrt(1.0 - (xpupil ** 2 + ypupil ** 2) * (self.na ** 2) / (self.refcov ** 2))
        costhetaimm = torch.sqrt(1.0 - (xpupil ** 2 + ypupil ** 2) * (self.na ** 2) / (self.refimm ** 2))
        fresnelpmedcov = 2 * self.refmed * costhetamed / (self.refmed * costhetacov + self.refcov * costhetamed)
        fresnelsmedcov = 2 * self.refmed * costhetamed / (self.refmed * costhetamed + self.refcov * costhetacov)
        fresnelpcovimm = 2 * self.refcov * costhetacov / (self.refcov * costhetaimm + self.refimm * costhetacov)
        fresnelscovimm = 2 * self.refcov * costhetacov / (self.refcov * costhetacov + self.refimm * costhetaimm)
        fresnelp = fresnelpmedcov * fresnelpcovimm
        fresnels = fresnelsmedcov * fresnelscovimm

        # apodization
        # apod = 1 / torch.sqrt(costhetaimm)  # previous version, for the simulated test dataset, should be deprecated
        # apod = 1 / torch.sqrt(costhetamed)
        apod = torch.sqrt(costhetaimm) / costhetamed  # Sjoerd Stallinga version

        # define aperture
        aperturemask = torch.where((xpupil ** 2 + ypupil ** 2).real < 1.0, 1.0, 0.0)
        self.amplitude = aperturemask * apod

        # setting of vectorial functions
        phi = torch.atan2(torch.real(ypupil), torch.real(xpupil))
        cosphi = torch.cos(phi)
        sinphi = torch.sin(phi)
        costheta = costhetamed
        sintheta = torch.sqrt(1 - costheta ** 2)

        pvec = torch.empty([3, self.npupil, self.npupil], dtype=self.complex_type, device='cuda')
        pvec[0] = fresnelp * costheta * cosphi
        pvec[1] = fresnelp * costheta * sinphi
        pvec[2] = -fresnelp * sintheta
        svec = torch.empty([3, self.npupil, self.npupil], dtype=self.complex_type, device='cuda')
        svec[0] = -fresnels * sinphi
        svec[1] = fresnels * cosphi
        svec[2] = 0 * cosphi

        polarizationvector = torch.empty([2, 3, self.npupil, self.npupil], dtype=self.complex_type, device='cuda')
        polarizationvector[0,] = cosphi * pvec - sinphi * svec
        polarizationvector[1,] = sinphi * pvec + cosphi * svec

        self.wavevector = torch.empty([2, self.npupil, self.npupil], dtype=self.complex_type, device='cuda')
        self.wavevector[0] = 2 * np.pi * self.na / self.wavelength * xpupil
        self.wavevector[1] = 2 * np.pi * self.na / self.wavelength * ypupil
        self.wavevectorzimm = 2 * np.pi * self.refimm / self.wavelength * costhetaimm
        self.wavevectorzmed = 2 * np.pi * self.refmed / self.wavelength * costhetamed

        # calculate aberration function
        waberration = torch.zeros_like(xpupil, dtype=self.complex_type, device='cuda')
        normfac = torch.sqrt(
            2 * (self.zernike_mode[:, 0] + 1) / (1 + torch.where(self.zernike_mode[:, 1] == 0, 1.0, 0.0)))
        zernikecoefs_norm = self.zernike_coef * normfac
        allzernikes = self.get_zernike(self.zernike_mode, xpupil, ypupil)

        waberration += torch.sum(zernikecoefs_norm[:, None, None]*allzernikes, dim=0)
        waberration *= aperturemask
        self.zernike_phase = torch.exp(1j * 2 * np.pi * waberration / self.wavelength)

        self.pupilmatrix = torch.empty([2, 3, self.npupil, self.npupil], dtype=self.complex_type, device='cuda')
        self.pupilmatrix = self.amplitude[None, None] * self.zernike_phase[None, None] * polarizationvector

        # czt transform(fft the pupil)
        xrange = self.pixel_size_xy[0] * self.psf_size / 2
        yrange = self.pixel_size_xy[1] * self.psf_size / 2
        imagesizex = xrange * self.na / self.wavelength
        imagesizey = yrange * self.na / self.wavelength

        # calculate the auxiliary vectors for chirp-z, pixelsize_xy should be inverse to match row and column
        self.ax, self.bx, self.dx = self.prechirpz(pupil_size, imagesizey, self.npupil, self.psf_size)
        self.ay, self.by, self.dy = self.prechirpz(pupil_size, imagesizex, self.npupil, self.psf_size)

        # calculate intensity normalization function using the PSF at focus
        fieldmatrix_norm = torch.empty([2, 3, self.psf_size, self.psf_size], dtype=self.complex_type, device='cuda')
        Pupilfunction_norm = self.amplitude[None, None, None] * polarizationvector[:, :, None]
        inter_image_norm = torch.transpose(self.czt_parallel(Pupilfunction_norm, self.ax, self.bx, self.dx), -1, -2)
        fieldmatrix_norm = torch.transpose(self.czt_parallel(inter_image_norm, self.ay, self.by, self.dy), -1, -2)

        int_focus = torch.zeros([self.psf_size, self.psf_size], dtype=self.data_type, device='cuda')
        int_focus += 1 / 3 * torch.sum(torch.abs(fieldmatrix_norm) ** 2, dim=(0, 1, 2))
        self.norm_intensity = torch.sum(int_focus)

    def _pre_compute(self):
        """
        Compute the common intermediate variables in advance, this can save time for PSFs simulation
        """
        # pupil radius (in diffraction units) and pupil coordinate sampling
        pupil_size = 1.0
        dxypupil = 2 * pupil_size / self.npupil
        xypupil = torch.arange(-pupil_size + dxypupil / 2, pupil_size, dxypupil, device='cuda', dtype=self.data_type)
        [xpupil, ypupil] = torch.meshgrid(xypupil, xypupil, indexing='ij')
        ypupil = torch.complex(ypupil, torch.zeros_like(ypupil))
        xpupil = torch.complex(xpupil, torch.zeros_like(xpupil))

        # calculation of relevant Fresnel-coefficients for the interfaces
        costhetamed = torch.sqrt(1.0 - (xpupil ** 2 + ypupil ** 2) * (self.na ** 2) / (self.refmed ** 2))
        costhetacov = torch.sqrt(1.0 - (xpupil ** 2 + ypupil ** 2) * (self.na ** 2) / (self.refcov ** 2))
        costhetaimm = torch.sqrt(1.0 - (xpupil ** 2 + ypupil ** 2) * (self.na ** 2) / (self.refimm ** 2))
        fresnelpmedcov = 2 * self.refmed * costhetamed / (self.refmed * costhetacov + self.refcov * costhetamed)
        fresnelsmedcov = 2 * self.refmed * costhetamed / (self.refmed * costhetamed + self.refcov * costhetacov)
        fresnelpcovimm = 2 * self.refcov * costhetacov / (self.refcov * costhetaimm + self.refimm * costhetacov)
        fresnelscovimm = 2 * self.refcov * costhetacov / (self.refcov * costhetacov + self.refimm * costhetaimm)
        fresnelp = fresnelpmedcov * fresnelpcovimm
        fresnels = fresnelsmedcov * fresnelscovimm

        # apodization
        # apod = 1 / torch.sqrt(costhetaimm)  # previous version, for the simulated test dataset, should be deprecated
        # apod = 1 / torch.sqrt(costhetamed)
        apod = torch.sqrt(costhetaimm) / costhetamed  # Sjoerd Stallinga version

        # define aperture
        aperturemask = torch.where((xpupil ** 2 + ypupil ** 2).real < 1.0, 1.0, 0.0)
        self.amplitude = aperturemask * apod

        # setting of vectorial functions
        phi = torch.atan2(torch.real(ypupil), torch.real(xpupil))
        cosphi = torch.cos(phi)
        sinphi = torch.sin(phi)
        costheta = costhetamed
        sintheta = torch.sqrt(1 - costheta ** 2)

        pvec = torch.empty([3, self.npupil, self.npupil], dtype=self.complex_type, device='cuda')
        pvec[0] = fresnelp * costheta * cosphi
        pvec[1] = fresnelp * costheta * sinphi
        pvec[2] = -fresnelp * sintheta
        svec = torch.empty([3, self.npupil, self.npupil], dtype=self.complex_type, device='cuda')
        svec[0] = -fresnels * sinphi
        svec[1] = fresnels * cosphi
        svec[2] = 0 * cosphi

        self.polarizationvector = torch.empty([2, 3, self.npupil, self.npupil], dtype=self.complex_type, device='cuda')
        self.polarizationvector[0,] = cosphi * pvec - sinphi * svec
        self.polarizationvector[1,] = sinphi * pvec + cosphi * svec

        self.wavevector = torch.empty([2, self.npupil, self.npupil], dtype=self.complex_type, device='cuda')
        self.wavevector[0] = 2 * np.pi * self.na / self.wavelength * xpupil
        self.wavevector[1] = 2 * np.pi * self.na / self.wavelength * ypupil
        self.wavevectorzimm = 2 * np.pi * self.refimm / self.wavelength * costhetaimm
        self.wavevectorzmed = 2 * np.pi * self.refmed / self.wavelength * costhetamed

        # calculate aberration function
        normfac = torch.sqrt(
            2 * (self.zernike_mode[:, 0] + 1) / (1 + torch.where(self.zernike_mode[:, 1] == 0, 1.0, 0.0)))
        self.allzernikes = self.get_zernike(self.zernike_mode, xpupil, ypupil) * normfac[:, None, None] * aperturemask[None]

        # czt transform(fft the pupil)
        xrange = self.pixel_size_xy[0] * self.psf_size / 2
        yrange = self.pixel_size_xy[1] * self.psf_size / 2
        imagesizex = xrange * self.na / self.wavelength
        imagesizey = yrange * self.na / self.wavelength

        # calculate the auxiliary vectors for chirp-z, pixelsize_xy should be inverse to match row and column
        self.ax, self.bx, self.dx = self.prechirpz(pupil_size, imagesizey, self.npupil, self.psf_size)
        self.ay, self.by, self.dy = self.prechirpz(pupil_size, imagesizex, self.npupil, self.psf_size)

        # calculate intensity normalization function using the PSF at focus
        pupilfunction_norm = self.amplitude[None, None, None] * self.polarizationvector[:, :, None]
        inter_image_norm = torch.transpose(self.czt_parallel(pupilfunction_norm, self.ax, self.bx, self.dx), -1, -2)
        fieldmatrix_norm = torch.transpose(self.czt_parallel(inter_image_norm, self.ay, self.by, self.dy), -1, -2)

        int_focus = torch.zeros([self.psf_size, self.psf_size], dtype=self.data_type, device='cuda')
        int_focus += 1 / 3 * torch.sum(torch.abs(fieldmatrix_norm) ** 2, dim=(0, 1, 2))
        self.norm_intensity = torch.sum(int_focus)

    @deprecated(reason="the same as matlab code, using for loop is slow")
    def simulate_v1(self, x, y, z, photons, objstage=None):
        """
        Run the simulation to generate the vector PSFs with the given positions

        Args:
            x (torch.Tensor): x positions of the PSFs, unit nm
            y (torch.Tensor): y positions of the PSFs, unit nm
            z (torch.Tensor): z positions of the PSFs, unit nm
            photons (torch.Tensor): photon counts of the PSFs, unit photons
            objstage (torch.Tensor): objective stage positions relative to the cover-slip (0),
                the closer to the sample, the smaller this value is (-), unit nm

        Returns:
            torch.Tensor: PSFs, unit photons
        """

        n_mol = x.shape[0]
        objstage = torch.zeros(x.shape[0], dtype=self.data_type, device='cuda') if objstage is None else objstage

        field_matrix = torch.empty([2, 3, n_mol, self.psf_size, self.psf_size],
                                   dtype=self.complex_type, device='cuda')
        for jz in range(n_mol):
            # xyz induced phase, x,y should be inverse to match the column, row
            if z[jz] + self.zemit0 >= 0:
                phase_xyz = -y[jz] * self.wavevector[0] - x[jz] * self.wavevector[1] + \
                            (z[jz] + self.zemit0) * self.wavevectorzmed
                position_phase = torch.exp(1j * (phase_xyz + (objstage[jz] + self.objstage0) *
                                                 self.wavevectorzimm))
            else:
                # print("warning! the emitter's position may not have physical meaning")
                phase_xyz = -y[jz] * self.wavevector[0] - x[jz] * self.wavevector[1]
                position_phase = torch.exp(1j * (phase_xyz + (objstage[jz] + self.objstage0 + z[jz]
                                                              + self.zemit0) * self.wavevectorzimm))
            for itel in range(2):
                for jtel in range(3):
                    pupil_tmp = position_phase * self.pupilmatrix[itel, jtel]
                    inter_image = torch.transpose(self.czt(pupil_tmp, self.ay, self.by, self.dy), 1, 0)
                    field_matrix[itel, jtel, jz] = torch.transpose(self.czt(inter_image, self.ax, self.bx, self.dx), 1,
                                                                   0)

        psfs_out = torch.zeros([n_mol, self.psf_size, self.psf_size], device='cuda', dtype=torch.float32)
        for jz in range(n_mol):
            for jtel in range(3):
                for itel in range(2):
                    psfs_out[jz, :, :] += 1 / 3 * (torch.abs(field_matrix[itel, jtel, jz])) ** 2

        psfs_out /= self.norm_intensity

        # otf rescale
        if self.otf_rescale_xy[0] or self.otf_rescale_xy[1] != 0:
            psfs_out = self.otf_rescale(psfdata=psfs_out, sigma_xy=self.otf_rescale_xy)

        # normalize the psf to 1, then multiply with the photon number
        # psfs_out /= psfs_out.sum(-1).sum(-1)[:, None, None]
        psfs_out *= photons[:, None, None]

        return psfs_out

    @deprecated(reason="parallel version of v1, but not compatible with simulate v3, where zernike phase "
                       "is not computed in advance")
    def simulate_v2(self, x, y, z, photons, objstage=None):
        """
        Run the simulation to generate the vector PSFs with the given positions

        Args:
            x (torch.Tensor): x positions of the PSFs, unit nm
            y (torch.Tensor): y positions of the PSFs, unit nm
            z (torch.Tensor): z positions of the PSFs, unit nm
            photons (torch.Tensor): photon counts of the PSFs, unit photons
            objstage (torch.Tensor): objective stage positions relative to the cover-slip (0),
                the closer to the sample, the smaller this value is (-), unit nm

        Returns:
            torch.Tensor: PSFs, unit photons
        """

        n_mol = x.shape[0]
        if n_mol == 0:
            return torch.zeros([0, self.psf_size, self.psf_size], device='cuda', dtype=torch.float32)

        objstage = torch.zeros(x.shape[0], dtype=self.data_type, device='cuda') if objstage is None else objstage

        # batch_size in parallel to save GPU memory
        slice_list = []
        batch_size = 100
        for i in np.arange(0, n_mol, batch_size):
            slice_list.append(slice(i, min(i + batch_size, n_mol)))

        psfs_out = torch.zeros([n_mol, self.psf_size, self.psf_size], device='cuda', dtype=torch.float32)
        for slice_tmp in slice_list:
            length_tmp = slice_tmp.stop - slice_tmp.start
            position_phase = torch.empty([length_tmp, self.npupil, self.npupil], dtype=self.complex_type, device='cuda')

            idx = torch.where(z[slice_tmp] + self.zemit0 >= 0)[0]
            phase_xyz_tmp = -y[slice_tmp][idx][:, None, None] * self.wavevector[0][None] - x[slice_tmp][idx][:, None, None] * \
                            self.wavevector[1][None] + \
                            (z[slice_tmp][idx] + self.zemit0)[:, None, None] * self.wavevectorzmed[None]
            position_phase[idx, :, :] = torch.exp(
                1j * (phase_xyz_tmp + (objstage[slice_tmp][idx][:, None, None] + self.objstage0) *
                      self.wavevectorzimm[None]))

            idx = torch.where(z[slice_tmp] + self.zemit0 < 0)[0]
            phase_xyz_tmp = -y[slice_tmp][idx][:, None, None] * self.wavevector[0][None] - x[slice_tmp][idx][:, None, None] * \
                            self.wavevector[1][None]
            position_phase[idx, :, :] = torch.exp(
                1j * (phase_xyz_tmp + (objstage[slice_tmp][idx][:, None, None] + self.objstage0 + z[slice_tmp][idx][:, None, None]
                                       + self.zemit0) * self.wavevectorzimm[None]))

            pupil_tmp = position_phase[None, None] * self.pupilmatrix[:, :, None]
            inter_image = torch.transpose(self.czt_parallel(pupil_tmp, self.ay, self.by, self.dy), -1, -2)
            field_matrix = torch.transpose(self.czt_parallel(inter_image, self.ax, self.bx, self.dx), -1, -2)

            psfs_out[slice_tmp] += 1 / 3 * torch.sum((torch.abs(field_matrix[:, :])) ** 2, dim=(0, 1))

        # # all in parallel, but may cause GPU memory overflow
        # position_phase = torch.empty([n_mol, self.npupil, self.npupil], dtype=self.complex_type, device='cuda')
        #
        # idx = torch.where(z + self.zemit0 >= 0)[0]
        # phase_xyz_tmp = -y[idx][:, None, None] * self.wavevector[0][None] - x[idx][:, None, None] * self.wavevector[1][None] + \
        #                 (z[idx] + self.zemit0)[:, None, None] * self.wavevectorzmed[None]
        # position_phase[idx, :, :] = torch.exp(1j * (phase_xyz_tmp + (objstage[idx][:, None, None] + self.objstage0) *
        #                                             self.wavevectorzimm[None]))
        #
        # idx = torch.where(z + self.zemit0 < 0)[0]
        # phase_xyz_tmp = -y[idx][:, None, None] * self.wavevector[0][None] - x[idx][:, None, None] * self.wavevector[1][None]
        # position_phase[idx, :, :] = torch.exp(1j * (phase_xyz_tmp + (objstage[idx][:, None, None] + self.objstage0 + z[idx][:, None, None]
        #                                                              + self.zemit0) * self.wavevectorzimm[None]))
        #
        # pupil_tmp = position_phase[None, None] * self.pupilmatrix[:, :, None]
        # inter_image = torch.transpose(self.czt_parallel(pupil_tmp, self.ay, self.by, self.dy), -1, -2)
        # field_matrix = torch.transpose(self.czt_parallel(inter_image, self.ax, self.bx, self.dx), -1, -2)
        #
        # psfs_out = torch.zeros([n_mol, self.psf_size, self.psf_size], device='cuda', dtype=torch.float32)
        # psfs_out += 1 / 3 * torch.sum((torch.abs(field_matrix[:, :])) ** 2, dim=(0, 1))

        # intensity normalization
        psfs_out /= self.norm_intensity

        # otf rescale
        if self.otf_rescale_xy[0] or self.otf_rescale_xy[1] != 0:
            psfs_out = self.otf_rescale(psfdata=psfs_out, sigma_xy=self.otf_rescale_xy)

        # normalize the psf to 1, then multiply with the photon number
        # psfs_out /= psfs_out.sum(-1).sum(-1)[:, None, None]
        psfs_out *= photons[:, None, None]

        return psfs_out

    def simulate(self, x, y, z, photons, objstage=None, zernike_coefs=None):
        """
        Run the simulation to generate the vector PSFs with the given positions

        Args:
            x (torch.Tensor): x positions of the PSFs, unit nm
            y (torch.Tensor): y positions of the PSFs, unit nm
            z (torch.Tensor): z positions of the PSFs, unit nm
            photons (torch.Tensor): photon counts of the PSFs, unit photons
            objstage (torch.Tensor): objective stage positions relative to the cover-slip (0),
                the closer to the sample, the smaller this value is (-), unit nm
            zernike_coefs (torch.Tensor or None): if not None, each psf can be assigned a different zernike
                coefficients from this array with shape (npsf, 21), otherwise use the common class
                property self.zernike_coef, unit nm

        Returns:
            torch.Tensor: PSFs, unit photons
        """

        n_mol = x.shape[0]
        if n_mol == 0:
            return torch.zeros([0, self.psf_size, self.psf_size], device='cuda', dtype=torch.float32)

        objstage = torch.zeros(x.shape[0], dtype=self.data_type, device='cuda') if objstage is None else objstage

        if zernike_coefs is None:
            zernike_phase = torch.exp(1j * 2 * np.pi *
                                      torch.sum(self.zernike_coef[:, None, None] * self.allzernikes, dim=0)
                                      / self.wavelength)
            pupilmatrix = (self.amplitude[None, None] *
                           zernike_phase[None, None] *
                           self.polarizationvector)[:, :, None]

        # batch_size in parallel to save GPU memory
        slice_list = []
        batch_size = 100
        for i in np.arange(0, n_mol, batch_size):
            slice_list.append(slice(i, min(i + batch_size, n_mol)))

        psfs_out = torch.zeros([n_mol, self.psf_size, self.psf_size], device='cuda', dtype=torch.float32)
        for slice_tmp in slice_list:
            if zernike_coefs is not None:
                zernike_phase = torch.exp(1j * 2 * np.pi *
                                          torch.sum(zernike_coefs[slice_tmp, :, None, None] * self.allzernikes[None], dim=1)
                                          / self.wavelength)
                pupilmatrix = (zernike_phase[None, None] *
                               self.polarizationvector[:, :, None] *
                               self.amplitude[None, None, None])

            length_tmp = slice_tmp.stop - slice_tmp.start
            position_phase = torch.empty([length_tmp, self.npupil, self.npupil], dtype=self.complex_type, device='cuda')

            idx = torch.where(z[slice_tmp] + self.zemit0 >= 0)[0]
            phase_xyz_tmp = -y[slice_tmp][idx][:, None, None] * self.wavevector[0][None] - x[slice_tmp][idx][:, None, None] * \
                            self.wavevector[1][None] + \
                            (z[slice_tmp][idx] + self.zemit0)[:, None, None] * self.wavevectorzmed[None]
            position_phase[idx, :, :] = torch.exp(
                1j * (phase_xyz_tmp + (objstage[slice_tmp][idx][:, None, None] + self.objstage0) *
                      self.wavevectorzimm[None]))

            idx = torch.where(z[slice_tmp] + self.zemit0 < 0)[0]
            phase_xyz_tmp = -y[slice_tmp][idx][:, None, None] * self.wavevector[0][None] - x[slice_tmp][idx][:, None, None] * \
                            self.wavevector[1][None]
            position_phase[idx, :, :] = torch.exp(
                1j * (phase_xyz_tmp + (objstage[slice_tmp][idx][:, None, None] + self.objstage0 + z[slice_tmp][idx][:, None, None]
                                       + self.zemit0) * self.wavevectorzimm[None]))

            pupil_tmp = position_phase[None, None] * pupilmatrix
            inter_image = torch.transpose(self.czt_parallel(pupil_tmp, self.ay, self.by, self.dy), -1, -2)
            field_matrix = torch.transpose(self.czt_parallel(inter_image, self.ax, self.bx, self.dx), -1, -2)

            psfs_out[slice_tmp] += 1 / 3 * torch.sum((torch.abs(field_matrix[:, :])) ** 2, dim=(0, 1))

        # intensity normalization by focus
        psfs_out /= self.norm_intensity

        # # normalize by themselves
        # norm_factor = psfs_out.sum(dim=(-1, -2))
        # psfs_out /= norm_factor[:, None, None]

        # otf rescale
        if self.otf_rescale_xy[0] or self.otf_rescale_xy[1]:
            psfs_out = self.otf_rescale(psfdata=psfs_out, sigma_xy=self.otf_rescale_xy)

        # multiply with the photon number
        psfs_out *= photons[:, None, None]

        return psfs_out

    def compute_crlb(self, x, y, z, photons, bgs):
        """
        Calculate the CRLB of this PSF model at give positions, photons and backgrounds.

        Args:
            x (torch.Tensor): x positions of the PSFs, unit nm
            y (torch.Tensor): y positions of the PSFs, unit nm
            z (torch.Tensor): z positions of the PSFs, unit nm
            photons (torch.Tensor): photon counts of the PSFs, unit photons
            bgs (torch.Tensor): background counts of the PSFs, unit photons

        Returns:
            (torch.Tensor,torch.Tensor): CRLB xyz (nPSFs, 3) and model PSFs, unit nm and photons
        """

        n_mol = x.shape[0]

        # calculate the derivatives
        [dudt, model] = self._compute_derivative(x, y, z, photons, bgs)
        # dudt = self._torch_jacobian_derivative(x, y, z, photons, bgs)

        # calculate hessian matrix, here only consider not shared parameters: x, y, z, photons, background
        num_pars = n_mol * 5
        t2 = 1 / model
        hessian = torch.zeros([num_pars, num_pars], device='cuda', dtype=self.data_type)
        for p1 in range(num_pars):
            temp1_zind = int(np.floor(p1 / 5))  # the index of data
            temp1_pind = int(p1 % 5)  # the index of parameter type
            temp1 = dudt[:, :, :, temp1_pind]  # the derivative of data concerning this parameter type
            for p2 in range(p1, num_pars):
                temp2_zind = int(np.floor(p2 / 5))
                temp2_pind = int(p2 % 5)
                temp2 = dudt[:, :, :, temp2_pind]

                # since all parameters are not shared, only the same molecule data makes sense
                # when multiply gradients of two parameters
                if temp1_zind == temp2_zind:
                    temp = t2[temp1_zind, :, :] * temp1[temp1_zind, :, :] * temp2[temp2_zind, :, :]
                    hessian[p1, p2] = torch.sum(temp)
                    hessian[p2, p1] = hessian[p1, p2]

        # calculate local fisher matrix and crlb
        xyz_crlb = torch.zeros([n_mol, 3], device='cuda')
        for j in range(n_mol):
            fisher_tmp = hessian[j * 5:j * 5 + 5, j * 5:j * 5 + 5]
            sqrt_crlb_tmp = torch.sqrt(torch.diag(torch.inverse(fisher_tmp)))
            xyz_crlb[j] = sqrt_crlb_tmp[0:3]

        return xyz_crlb, model

    def compute_crlb_mf(self, x, y, z, photons, bgs, attn_length):
        #todo: need test
        """
        Calculate the CRLB of this PSF model at give positions, photons and backgrounds.

        Args:
            x (torch.Tensor): x positions of the PSFs, unit nm
            y (torch.Tensor): y positions of the PSFs, unit nm
            z (torch.Tensor): z positions of the PSFs, unit nm
            photons (torch.Tensor): photon counts of the PSFs, unit photons
            bgs (torch.Tensor): background counts of the PSFs, unit photons
            attn_length (int): attention length of the network, used to multiply the Fisher matrix

        Returns:
            (torch.Tensor,torch.Tensor): CRLB xyz (nPSFs, 3) and model PSFs, unit nm and photons
        """

        n_mol = x.shape[0]

        # calculate the derivatives
        [dudt, model] = self._compute_derivative_parallel(x, y, z, photons, bgs)

        # calculate hessian matrix, here only consider not shared parameters: x, y, z, photons, background
        num_pars = n_mol * 5
        t2 = 1 / model
        hessian = torch.zeros([num_pars, num_pars], device='cuda', dtype=self.data_type)
        for p1 in range(num_pars):
            temp1_zind = int(np.floor(p1 / 5))  # the index of data
            temp1_pind = int(p1 % 5)  # the index of parameter type
            temp1 = dudt[:, :, :, temp1_pind]  # the derivative of data concerning this parameter type
            for p2 in range(p1, num_pars):
                temp2_zind = int(np.floor(p2 / 5))
                temp2_pind = int(p2 % 5)
                temp2 = dudt[:, :, :, temp2_pind]

                # since all parameters are not shared, only the same molecule data makes sense
                # when multiply gradients of two parameters
                if temp1_zind == temp2_zind:
                    temp = t2[temp1_zind, :, :] * temp1[temp1_zind, :, :] * temp2[temp2_zind, :, :]
                    hessian[p1, p2] = torch.sum(temp)
                    hessian[p2, p1] = hessian[p1, p2]

        # calculate local fisher matrix and crlb
        xyz_crlb = torch.zeros([n_mol, 3], device='cuda')
        for j in range(n_mol):
            fisher_tmp = hessian[j * 5:j * 5 + 5, j * 5:j * 5 + 5]
            fisher_tmp[:3, :3] *= attn_length
            sqrt_crlb_tmp = torch.sqrt(torch.diag(torch.inverse(fisher_tmp)))
            xyz_crlb[j] = sqrt_crlb_tmp[0:3]

        return xyz_crlb, model

    @deprecated(reason='the same as matlab code, using for loop is slow')
    def _compute_derivative_v1(self, x, y, z, photons, bgs):
        """
        Calculate the analytical derivatives of the PSFs at given parameters with respect to x,y,z,photons,bg
        """

        n_mol = x.shape[0]
        objstage = torch.zeros(x.shape[0], dtype=self.data_type, device='cuda')
        field_matrix = torch.empty([2, 3, n_mol, self.psf_size, self.psf_size],
                                   dtype=self.complex_type, device='cuda')
        field_matrix_ders = torch.empty([2, 3, n_mol, 3, self.psf_size, self.psf_size],
                                        dtype=self.complex_type, device='cuda')
        for jz in range(n_mol):
            # xyz induced phase
            if z[jz] + self.zemit0 >= 0:
                phase_xyz = -y[jz] * self.wavevector[0] - x[jz] * self.wavevector[1] + \
                            (z[jz] + self.zemit0) * self.wavevectorzmed
                position_phase = torch.exp(1j * (phase_xyz + (objstage[jz] + self.objstage0) *
                                                 self.wavevectorzimm))
            else:
                # print("warning! the emitter's position may not have physical meaning")
                phase_xyz = -y[jz] * self.wavevector[0] - x[jz] * self.wavevector[1]
                position_phase = torch.exp(1j * (phase_xyz + (objstage[jz] + self.objstage0 + z[jz]
                                                              + self.zemit0) * self.wavevectorzimm))
            for itel in range(2):
                for jtel in range(3):
                    pupil_tmp = position_phase * self.pupilmatrix[itel, jtel]
                    inter_image = torch.transpose(self.czt(pupil_tmp, self.ay, self.by, self.dy), 1, 0)
                    field_matrix[itel, jtel, jz] = torch.transpose(self.czt(inter_image, self.ax, self.bx, self.dx), 1,
                                                                   0)
                    # derivatives with respect to x,y,z
                    pupilfunction_x = -1j * self.wavevector[1] * position_phase * self.pupilmatrix[itel, jtel]
                    inter_image_x = torch.transpose(self.czt(pupilfunction_x, self.ay, self.by, self.dy), 1, 0)
                    field_matrix_ders[itel, jtel, jz, 0] = torch.transpose(self.czt(inter_image_x, self.ax,
                                                                                    self.bx, self.dx), 1, 0)

                    pupilfunction_y = -1j * self.wavevector[0] * position_phase * self.pupilmatrix[itel, jtel]
                    inter_image_y = torch.transpose(self.czt(pupilfunction_y, self.ay, self.by, self.dy), 1, 0)
                    field_matrix_ders[itel, jtel, jz, 1] = torch.transpose(self.czt(inter_image_y, self.ax,
                                                                                    self.bx, self.dx), 1, 0)

                    if z[jz] + self.zemit0 >= 0:
                        pupilfunction_z = 1j * self.wavevectorzmed * position_phase * self.pupilmatrix[itel, jtel]
                    else:
                        pupilfunction_z = 1j * self.wavevectorzimm * position_phase * self.pupilmatrix[itel, jtel]
                    inter_image_z = torch.transpose(self.czt(pupilfunction_z, self.ay, self.by, self.dy), 1, 0)
                    field_matrix_ders[itel, jtel, jz, 2] = torch.transpose(self.czt(inter_image_z, self.ax,
                                                                                    self.bx, self.dx), 1, 0)

        psfs = torch.zeros([n_mol, self.psf_size, self.psf_size], device='cuda', dtype=self.data_type)
        psfs_ders = torch.zeros([n_mol, self.psf_size, self.psf_size, 3], device='cuda', dtype=self.data_type)
        for jz in range(n_mol):
            for jtel in range(3):
                for itel in range(2):
                    psfs[jz, :, :] += 1 / 3 * (torch.abs(field_matrix[itel, jtel, jz])) ** 2
                    for jder in range(3):
                        psfs_ders[jz, :, :, jder] = psfs_ders[jz, :, :, jder] + 2 / 3 * \
                                                    torch.real(torch.conj(field_matrix[itel, jtel, jz]) *
                                                               field_matrix_ders[itel, jtel, jz, jder])
        psfs /= self.norm_intensity
        psfs_ders /= self.norm_intensity

        # otf rescale
        if self.otf_rescale_xy[0] or self.otf_rescale_xy[1] != 0:
            psfs = self.otf_rescale(psfdata=psfs, sigma_xy=self.otf_rescale_xy)
            psfs_ders[:, :, :, 0] = self.otf_rescale(psfdata=psfs_ders[:, :, :, 0], sigma_xy=self.otf_rescale_xy)
            psfs_ders[:, :, :, 1] = self.otf_rescale(psfdata=psfs_ders[:, :, :, 1], sigma_xy=self.otf_rescale_xy)
            psfs_ders[:, :, :, 2] = self.otf_rescale(psfdata=psfs_ders[:, :, :, 2], sigma_xy=self.otf_rescale_xy)

        psfs_out = gpu(psfs * photons[:, None, None] + bgs[:, None, None])
        ders_out = torch.zeros([n_mol, self.psf_size, self.psf_size, 5], device='cuda', dtype=self.data_type)
        ders_out[:, :, :, 0:3] = psfs_ders * photons[:, None, None, None]
        ders_out[:, :, :, 3] = psfs
        ders_out[:, :, :, 4] = torch.ones_like(psfs)
        return ders_out, psfs_out

    @deprecated(reason='parallel version of v1, but not compatible with the _pre_compute_v3 and simulate_v3')
    def _compute_derivative_v2(self, x, y, z, photons, bgs):
        """
        Calculate the analytical derivatives of the PSFs at given parameters with respect to x,y,z,photons,bg
        """

        n_mol = x.shape[0]
        objstage = torch.zeros(x.shape[0], dtype=self.data_type, device='cuda')

        # batch_size in parallel to save GPU memory
        slice_list = []
        batch_size = 100
        for i in np.arange(0, n_mol, batch_size):
            slice_list.append(slice(i, min(i + batch_size, n_mol)))

        psfs = torch.zeros([n_mol, self.psf_size, self.psf_size], device='cuda', dtype=self.data_type)
        psfs_ders = torch.zeros([n_mol, self.psf_size, self.psf_size, 3], device='cuda', dtype=self.data_type)
        for slice_tmp in slice_list:
            length_tmp = slice_tmp.stop - slice_tmp.start
            position_phase = torch.empty([length_tmp, self.npupil, self.npupil], dtype=self.complex_type, device='cuda')

            idx_0 = torch.where(z[slice_tmp] + self.zemit0 >= 0)[0]
            phase_xyz_tmp = -y[slice_tmp][idx_0][:, None, None] * self.wavevector[0][None] - x[slice_tmp][idx_0][:, None, None] * \
                            self.wavevector[1][None] + \
                            (z[slice_tmp][idx_0] + self.zemit0)[:, None, None] * self.wavevectorzmed[None]
            position_phase[idx_0, :, :] = torch.exp(
                1j * (phase_xyz_tmp + (objstage[slice_tmp][idx_0][:, None, None] + self.objstage0) *
                      self.wavevectorzimm[None]))

            idx_1 = torch.where(z[slice_tmp] + self.zemit0 < 0)[0]
            phase_xyz_tmp = -y[slice_tmp][idx_1][:, None, None] * self.wavevector[0][None] - x[slice_tmp][idx_1][:, None, None] * \
                            self.wavevector[1][None]
            position_phase[idx_1, :, :] = torch.exp(
                1j * (phase_xyz_tmp + (objstage[slice_tmp][idx_1][:, None, None] + self.objstage0 + z[slice_tmp][idx_1][:, None, None]
                                       + self.zemit0) * self.wavevectorzimm[None]))

            pupil_tmp = position_phase[None, None] * self.pupilmatrix[:, :, None]
            inter_image = torch.transpose(self.czt_parallel(pupil_tmp, self.ay, self.by, self.dy), -1, -2)
            field_matrix = torch.transpose(self.czt_parallel(inter_image, self.ax, self.bx, self.dx), -1, -2)
            psfs[slice_tmp] += 1 / 3 * torch.sum((torch.abs(field_matrix[:, :])) ** 2, dim=(0, 1))

            # derivatives with respect to x,y,z
            pupil_tmp_x = -1j * self.wavevector[1] * position_phase[None, None] * self.pupilmatrix[:, :, None]
            inter_image_x = torch.transpose(self.czt_parallel(pupil_tmp_x, self.ay, self.by, self.dy), -1, -2)
            field_matrix_x = torch.transpose(self.czt_parallel(inter_image_x, self.ax, self.bx, self.dx), -1, -2)
            psfs_ders[slice_tmp, :, :, 0] += 2 / 3 * torch.sum(torch.real(torch.conj(field_matrix) * field_matrix_x),
                                                               dim=(0, 1))

            pupil_tmp_y = -1j * self.wavevector[0] * position_phase[None, None] * self.pupilmatrix[:, :, None]
            inter_image_y = torch.transpose(self.czt_parallel(pupil_tmp_y, self.ay, self.by, self.dy), -1, -2)
            field_matrix_y = torch.transpose(self.czt_parallel(inter_image_y, self.ax, self.bx, self.dx), -1, -2)
            psfs_ders[slice_tmp, :, :, 1] += 2 / 3 * torch.sum(torch.real(torch.conj(field_matrix) * field_matrix_y),
                                                               dim=(0, 1))

            pupil_tmp_z = torch.empty([2, 3, length_tmp, self.npupil, self.npupil], dtype=self.complex_type, device='cuda')
            pupil_tmp_z[:, :, idx_0] = 1j * self.wavevectorzmed * position_phase[idx_0] * self.pupilmatrix[:, :, None]
            pupil_tmp_z[:, :, idx_1] = 1j * self.wavevectorzimm * position_phase[idx_1] * self.pupilmatrix[:, :, None]
            inter_image_z = torch.transpose(self.czt_parallel(pupil_tmp_z, self.ay, self.by, self.dy), -1, -2)
            field_matrix_z = torch.transpose(self.czt_parallel(inter_image_z, self.ax, self.bx, self.dx), -1, -2)
            psfs_ders[slice_tmp, :, :, 2] += 2 / 3 * torch.sum(torch.real(torch.conj(field_matrix) * field_matrix_z),
                                                               dim=(0, 1))
        psfs /= self.norm_intensity
        psfs_ders /= self.norm_intensity

        # otf rescale
        if self.otf_rescale_xy[0] or self.otf_rescale_xy[1] != 0:
            psfs = self.otf_rescale(psfdata=psfs, sigma_xy=self.otf_rescale_xy)
            psfs_ders[:, :, :, 0] = self.otf_rescale(psfdata=psfs_ders[:, :, :, 0], sigma_xy=self.otf_rescale_xy)
            psfs_ders[:, :, :, 1] = self.otf_rescale(psfdata=psfs_ders[:, :, :, 1], sigma_xy=self.otf_rescale_xy)
            psfs_ders[:, :, :, 2] = self.otf_rescale(psfdata=psfs_ders[:, :, :, 2], sigma_xy=self.otf_rescale_xy)

        psfs_out = gpu(psfs * photons[:, None, None] + bgs[:, None, None])
        ders_out = torch.zeros([n_mol, self.psf_size, self.psf_size, 5], device='cuda', dtype=self.data_type)
        ders_out[:, :, :, 0:3] = psfs_ders * photons[:, None, None, None]
        ders_out[:, :, :, 3] = psfs
        ders_out[:, :, :, 4] = torch.ones_like(psfs)

        return ders_out, psfs_out

    def _compute_derivative(self, x, y, z, photons, bgs):
        """
        Calculate the analytical derivatives of the PSFs at given parameters with respect to x,y,z,photons,bg
        """

        n_mol = x.shape[0]
        objstage = torch.zeros(x.shape[0], dtype=self.data_type, device='cuda')

        zernike_phase = torch.exp(1j * 2 * np.pi *
                                  torch.sum(self.zernike_coef[:, None, None]*self.allzernikes, dim=0)
                                  / self.wavelength)
        pupilmatrix = (self.amplitude[None, None] *
                       zernike_phase[None, None] *
                       self.polarizationvector)[:, :, None]

        # batch_size in parallel to save GPU memory
        slice_list = []
        batch_size = 100
        for i in np.arange(0, n_mol, batch_size):
            slice_list.append(slice(i, min(i + batch_size, n_mol)))

        psfs = torch.zeros([n_mol, self.psf_size, self.psf_size], device='cuda', dtype=self.data_type)
        psfs_ders = torch.zeros([n_mol, self.psf_size, self.psf_size, 3], device='cuda', dtype=self.data_type)
        for slice_tmp in slice_list:
            length_tmp = slice_tmp.stop - slice_tmp.start
            position_phase = torch.empty([length_tmp, self.npupil, self.npupil], dtype=self.complex_type, device='cuda')

            idx_0 = torch.where(z[slice_tmp] + self.zemit0 >= 0)[0]
            phase_xyz_tmp = -y[slice_tmp][idx_0][:, None, None] * self.wavevector[0][None] - x[slice_tmp][idx_0][:, None, None] * \
                            self.wavevector[1][None] + \
                            (z[slice_tmp][idx_0] + self.zemit0)[:, None, None] * self.wavevectorzmed[None]
            position_phase[idx_0, :, :] = torch.exp(
                1j * (phase_xyz_tmp + (objstage[slice_tmp][idx_0][:, None, None] + self.objstage0) *
                      self.wavevectorzimm[None]))

            idx_1 = torch.where(z[slice_tmp] + self.zemit0 < 0)[0]
            phase_xyz_tmp = -y[slice_tmp][idx_1][:, None, None] * self.wavevector[0][None] - x[slice_tmp][idx_1][:, None, None] * \
                            self.wavevector[1][None]
            position_phase[idx_1, :, :] = torch.exp(
                1j * (phase_xyz_tmp + (objstage[slice_tmp][idx_1][:, None, None] + self.objstage0 + z[slice_tmp][idx_1][:, None, None]
                                       + self.zemit0) * self.wavevectorzimm[None]))

            pupil_tmp = position_phase[None, None] * pupilmatrix
            inter_image = torch.transpose(self.czt_parallel(pupil_tmp, self.ay, self.by, self.dy), -1, -2)
            field_matrix = torch.transpose(self.czt_parallel(inter_image, self.ax, self.bx, self.dx), -1, -2)
            psfs[slice_tmp] += 1 / 3 * torch.sum((torch.abs(field_matrix[:, :])) ** 2, dim=(0, 1))

            # derivatives with respect to x,y,z
            pupil_tmp_x = -1j * self.wavevector[1] * position_phase[None, None] * pupilmatrix
            inter_image_x = torch.transpose(self.czt_parallel(pupil_tmp_x, self.ay, self.by, self.dy), -1, -2)
            field_matrix_x = torch.transpose(self.czt_parallel(inter_image_x, self.ax, self.bx, self.dx), -1, -2)
            psfs_ders[slice_tmp, :, :, 0] += 2 / 3 * torch.sum(torch.real(torch.conj(field_matrix) * field_matrix_x),
                                                               dim=(0, 1))

            pupil_tmp_y = -1j * self.wavevector[0] * position_phase[None, None] * pupilmatrix
            inter_image_y = torch.transpose(self.czt_parallel(pupil_tmp_y, self.ay, self.by, self.dy), -1, -2)
            field_matrix_y = torch.transpose(self.czt_parallel(inter_image_y, self.ax, self.bx, self.dx), -1, -2)
            psfs_ders[slice_tmp, :, :, 1] += 2 / 3 * torch.sum(torch.real(torch.conj(field_matrix) * field_matrix_y),
                                                               dim=(0, 1))

            pupil_tmp_z = torch.empty([2, 3, length_tmp, self.npupil, self.npupil], dtype=self.complex_type, device='cuda')
            pupil_tmp_z[:, :, idx_0] = 1j * self.wavevectorzmed * position_phase[idx_0] * pupilmatrix
            pupil_tmp_z[:, :, idx_1] = 1j * self.wavevectorzimm * position_phase[idx_1] * pupilmatrix
            inter_image_z = torch.transpose(self.czt_parallel(pupil_tmp_z, self.ay, self.by, self.dy), -1, -2)
            field_matrix_z = torch.transpose(self.czt_parallel(inter_image_z, self.ax, self.bx, self.dx), -1, -2)
            psfs_ders[slice_tmp, :, :, 2] += 2 / 3 * torch.sum(torch.real(torch.conj(field_matrix) * field_matrix_z),
                                                               dim=(0, 1))
        # normalize by focus intensity
        psfs /= self.norm_intensity
        psfs_ders /= self.norm_intensity

        # # normalize by themselves
        # norm_factor = psfs.sum(dim=(-1, -2))
        # psfs /= norm_factor[:, None, None]
        # psfs_ders /= norm_factor[:, None, None, None]

        # otf rescale
        if self.otf_rescale_xy[0] or self.otf_rescale_xy[1]:
            psfs = self.otf_rescale(psfdata=psfs, sigma_xy=self.otf_rescale_xy)
            psfs_ders[:, :, :, 0] = self.otf_rescale(psfdata=psfs_ders[:, :, :, 0], sigma_xy=self.otf_rescale_xy)
            psfs_ders[:, :, :, 1] = self.otf_rescale(psfdata=psfs_ders[:, :, :, 1], sigma_xy=self.otf_rescale_xy)
            psfs_ders[:, :, :, 2] = self.otf_rescale(psfdata=psfs_ders[:, :, :, 2], sigma_xy=self.otf_rescale_xy)

        psfs_out = gpu(psfs * photons[:, None, None] + bgs[:, None, None])
        ders_out = torch.zeros([n_mol, self.psf_size, self.psf_size, 5], device='cuda', dtype=self.data_type)
        ders_out[:, :, :, 0:3] = psfs_ders * photons[:, None, None, None]
        ders_out[:, :, :, 3] = psfs
        ders_out[:, :, :, 4] = torch.ones_like(psfs)

        return ders_out, psfs_out

    def _torch_jacobian_derivative(self, x, y, z, photons, bgs):
        """Calculate the derivatives of the PSFs at given parameters with respect to x,y,z,photons,bg
        using torch.autograd.functional.jacobian"""
        x.requires_grad = True
        y.requires_grad = True
        z.requires_grad = True
        photons.requires_grad = True
        bgs.requires_grad = True
        jacobian = torch.autograd.functional.jacobian(self.simulate_parallel, (x, y, z, photons))

        ders_out = torch.zeros([x.shape[0], self.psf_size, self.psf_size, 5], device='cuda', dtype=self.data_type)
        for i in range(len(jacobian)):
            for j in range(x.shape[0]):
                ders_out[j, :, :, i] = jacobian[i][j, :, :, j]
        ders_out[:, :, :, 4] = torch.ones([x.shape[0], self.psf_size, self.psf_size])

        return ders_out

    def optimize_crlb(self, x, y, z, photons, bgs, tolerance):
        """
        Optimize the CRLB of this PSF model with respect to zernike coefficients
        at give positions, photons and backgrounds. The instance should be initialized with req_grad=True

        Args:
            x (torch.Tensor): x positions of the PSFs, unit nm
            y (torch.Tensor): y positions of the PSFs, unit nm
            z (torch.Tensor): z positions of the PSFs, unit nm
            photons (torch.Tensor): photon counts of the PSFs, unit photon
            bgs (torch.Tensor): background counts of the PSFs, unit photon
            tolerance (float): stop criteria, the difference of CRLB between two iterations

        Returns:

        """

        n_mol = x.shape[0]
        crlb_optimizer = torch.optim.Adam([self.zernike_coef], lr=0.1)
        loss_1 = 1e6
        iter_num = 1
        # for iter_num in range(iterations):
        while True:
            xyz_crlb, model = self.compute_crlb(x, y, z, photons, bgs)
            crlb_3d_avg = torch.sum(xyz_crlb[:, 0]**2 + xyz_crlb[:, 1]**2 + xyz_crlb[:, 2]**2)/n_mol
            print(f"iter: {iter_num}, crlb_3d_avg: {crlb_3d_avg}")
            crlb_optimizer.zero_grad()
            crlb_3d_avg.backward()
            crlb_optimizer.step()
            self._pre_compute()
            if torch.abs((loss_1 - crlb_3d_avg) / loss_1) < tolerance:
                break
            loss_1 = crlb_3d_avg.detach()
            iter_num += 1
        print('CRLB optimization done')
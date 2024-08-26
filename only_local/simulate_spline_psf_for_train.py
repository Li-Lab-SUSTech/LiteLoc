import numpy as np
import torch

from utils.data_generator import DataGenerator

zernike_aber = np.array([2, -2, 0, 2, 2, 70, 3, -1, 0, 3, 1, 0, 4, 0, 0, 3, -3, 0, 3, 3, 0,
                         4, -2, 0, 4, 2, 0, 5, -1, 0, 5, 1, 0, 6, 0, 0, 4, -4, 0, 4, 4, 0,
                         5, -3, 0, 5, 3, 0, 6, -2, 0, 6, 2, 0, 7, 1, 0, 7, -1, 0, 8, 0, 0],
                        dtype=np.float32).reshape([21, 3])

zernike_mode = zernike_aber[:, 0:2]
zernike_coef = zernike_aber[:, 2]

camera_params = {'camera': 'sCMOS', 'em_gain': 1.0, 'surv_p': 0.5, 'margin_empty': 0.1,
                 'qe': 0.81, 'spurious_c': 0.002, 'sig_read': 1.61, 'e_per_adu': 0.47, 'baseline': 100.0}

psf_params = {'ph_scale': 7500,
              'Npixels': 51,
              'z_scale': 700,
              'bg': 135,
              'pixel_size_xy': [108, 108],
              'robust_training': False,
              'objstage0': -1000,
              'psf_size': 51,
              'NA': 1.5,
              'refmed': 1.406,
              'refcov': 1.524,
              'refimm': 1.518,
              'wavelength': 670,
              'psf': 'Astigmatism',
              'simulate_method': 'torch',
              'perlin_noise': True,
              'pn_factor': 0.5,
              'pn_res': 64,
              'zernike_mode': zernike_mode, 'zernike_coef': zernike_coef,
              'otf_rescale_xy': [0.5, 0.5],
              'Npupil': 64}

train_params = {'max_iters': 50,
                'interval': 500,
                'batch_size': 10,
                'ph_filt_thre': 500,
                'nvalid': 30,
                'em_per_frame': 40,
                'train_size': [128, 128],
                'z_scale': 700,
                'min_ph': 0.067,
                'results_path': "./training_model/LiteLoc_setfs_astig_local_loss_noweight/",
                'dataresume': False, 'netresume': False}

spline_params = {'calibration_file': 'Astigmatism_Crimson_645_beads_2um_50nm_2mw_roi_512_1_MMStack_Default.ome_3dcal.mat',
                 'psf_extent': [[0, 128], [0, 128], None], 'img_size': [128, 128],
                 'device_simulation': 'cuda', 'roi_size': None, 'roi_auto_center': False}


DataGen = DataGenerator(train_params, camera_params, psf_params)

locs, X, Y, Z, I, s_mask, xyzi_gt, S = DataGen.generate_batch(size=DataGen.batch_size, val=False, local_context=False)

molecule_tuple = tuple(s_mask.nonzero().transpose(1, 0))
xyz_px = (xyzi_gt[molecule_tuple[0], molecule_tuple[1], :3][:, [1, 0, 2]].cpu())
intensity = (xyzi_gt[molecule_tuple[0], molecule_tuple[1], 3] * psf_params['ph_scale']).cpu()
frame_ix = torch.squeeze(s_mask.nonzero()[:, 0]).cpu()

img_sim = DataGen.simulated_splinePSF(xyz_px, intensity, frame_ix, spline_params)


import numpy as np
import torch
import os
import scipy.io as scio
import pickle
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from utils.perlin_noise import *
from vector_psf.vectorpsf import VectorPSFTorch
from utils.help_utils import place_psfs, gpu, gpu_cpu_torch
from spline_psf.calibration_io import SMAPSplineCoefficient
from utils.help_utils import format_psf_model_params
from utils.compat_utils import get_device


class LocalizeDataset(Dataset):

    # initialization of the dataset
    def __init__(self, list_IDs, labels):
        self.list_IDs = list_IDs
        self.labels = labels

    # total number of samples in the dataset
    def __len__(self):
        return len(self.list_IDs)

    # sampling one example from the data
    def __getitem__(self, index):
        # select sample
        ID = self.list_IDs[index]
        # associated number of photons
        dict = self.labels[ID]
        Nphotons = dict['ints']
        xemit= dict['x_os']
        yemit= dict['y_os']
        z= dict['z']
        locs = dict['locs']
        s_mask = dict['s_mask']
        gt = dict['gt']
        simu_image = dict['simu_image']
        return xemit, yemit, z, locs, Nphotons, s_mask, gt, simu_image


class DataGenerator:

    def __init__(self, train_params, camera_params, psf_params, device=get_device()):
        self.path_train = train_params.result_path
        self.batch_size = train_params.batch_size
        self.valid_size = train_params.valid_frame_num
        self.num_particles = train_params.em_per_frame
        self.train_size_x = train_params.train_size[0]
        self.train_size_y = train_params.train_size[1]
        self.min_ph = train_params.photon_range[0] / train_params.photon_range[1]
        self.ph_scale = train_params.photon_range[1]
        self.nvalid_batches = int(np.ceil(train_params.valid_frame_num/train_params.batch_size))
        self.camera_params = camera_params
        self.perlin_noise = train_params.perlin_noise
        self.pn_factor = train_params.pn_factor
        self.pn_res = train_params.pn_res
        self.bg = train_params.bg
        self.psf_model = psf_params.simulate_method
        self.z_scale = psf_params.z_scale
        self.robust_training = False
        
        self.device = device
        
        if self.psf_model == 'vector' or self.psf_model == 'ui_psf':
            vector_params, zernike, objstage0, pixel_size_xy, zernike_init, robust_training = format_psf_model_params(psf_params)
            self.vector_params, self.zernike, self.objstage0, self.zernike_fit, self.robust_training = vector_params, zernike, objstage0, zernike_init, robust_training
            self.pixel_size_x = pixel_size_xy[0]
            self.pixel_size_y = pixel_size_xy[1]
            self.VectorPSF = VectorPSFTorch(self.vector_params, self.zernike, self.objstage0, device=self.device)
        elif self.psf_model == 'spline':
            self.spline_params = psf_params.spline_psf
            self.psf = SMAPSplineCoefficient(calib_file=self.spline_params.calibration_file).init_spline(
                xextent=self.spline_params.psf_extent[0],
                yextent=self.spline_params.psf_extent[1],
                img_shape=train_params.train_size,
                device='cuda' if self.device.type == 'cuda' else 'cpu',  # cuda or cpu, mps not support
                roi_size=None,
                roi_auto_center=None
            )
        else:
            print('\n***Input PSF model method name cannot be recognizable! Please check it!***\n')

    def gen_valid_data(self):

        os.makedirs(self.path_train, exist_ok=True)
        # print status
        print('=' * 50)
        print('Sampling examples for validation')
        print('=' * 50)

        labels_dict = {}

        # sample validation examples
        for i in range(self.nvalid_batches):
            # sample a training example
            while True:
                locs, X_os, Y_os, Z, I, s_mask, gt, S = self.generate_batch(self.batch_size, local_context=False)

                img_sim = self.simulate_image(s_mask, gt, S, torch.squeeze(X_os),
                                                      torch.squeeze(Y_os), torch.squeeze(Z), torch.squeeze(I), mode='eval')

                X_os = torch.squeeze(X_os)
                Y_os = torch.squeeze(Y_os)
                Z = torch.squeeze(Z)
                I = torch.squeeze(I)
                if locs.sum() > 0 and X_os.size():
                    labels_dict[str(i)] = {'locs': locs,
                                           'x_os': X_os, 'y_os': Y_os, 'z': Z, 'ints': I,
                                           'gt': gt, 's_mask': s_mask, 'simu_image': img_sim}
                    break

        # save all xyz's dictionary as a pickle file
        path_labels = self.path_train + 'validLabels.pickle'
        self.labels = labels_dict
        with open(path_labels, 'wb') as handle:
            pickle.dump(labels_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return self.get_valid_dataset()

    def gen_valid_data_deepstorm3d(self):

        os.makedirs(self.path_train, exist_ok=True)
        # print status
        print('=' * 50)
        print('Sampling examples for validation')
        print('=' * 50)

        labels_dict = {}

        # sample validation examples
        for i in range(self.nvalid_batches):
            # sample a training example
            while True:

                locs, X_os, Y_os, Z, I, s_mask, gt, xyzi_batch = self.DataGen.generate_batch_deepstorm3d(
                    self.params.Training.batch_size,
                    val=False)
                img_sim = self.DataGen.simulatedImg_torch(locs, X, Y, Z, I).reshape(
                    [self.params.Training.batch_size, 1, locs.shape[-2], locs.shape[-1]])

                X_os = torch.squeeze(X_os)
                Y_os = torch.squeeze(Y_os)
                Z = torch.squeeze(Z)
                I = torch.squeeze(I)
                if locs.sum() > 0 and X_os.size():
                    labels_dict[str(i)] = {'locs': locs,
                                           'x_os': X_os, 'y_os': Y_os, 'z': Z, 'ints': I,
                                           'gt': gt, 's_mask': s_mask, 'simu_image': img_sim}
                    break

        # save all xyz's dictionary as a pickle file
        path_labels = self.path_train + 'validLabels.pickle'
        self.labels = labels_dict
        with open(path_labels, 'wb') as handle:
            pickle.dump(labels_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return self.get_valid_dataset()

    def read_valid_file(self):
        with open( self.path_train +'validLabels.pickle', 'rb') as handle:
            self.labels = pickle.load(handle)
        return self.get_valid_dataset()

    def get_valid_dataset(self):
        ind_all = np.arange(0, self.nvalid_batches, 1)
        list_all = ind_all.tolist()
        list_IDs = [str(i) for i in list_all]

        # instantiate the data class and create a data loader for validation
        validation_set = LocalizeDataset(list_IDs, self.labels)
        validation_generator = DataLoader(validation_set, shuffle=False)
        return validation_generator

    def simulate_image(self, s_mask_all=None, xyzi_gt_all=None, S=None, xnm=None, ynm=None, Z=None, I=None, mode='train'):
        if self.psf_model == 'spline':
            molecule_tuple = tuple(s_mask_all.nonzero().transpose(1, 0))
            xyz_px = xyzi_gt_all[molecule_tuple[0], molecule_tuple[1], :3][:, [1, 0, 2]]
            xyz_px[:, 2] = xyz_px[:, 2] * self.z_scale
            xyz_px = xyz_px.cpu()
            intensity = (xyzi_gt_all[molecule_tuple[0], molecule_tuple[1], 3] * self.ph_scale).cpu()
            frame_ix = torch.squeeze(s_mask_all.nonzero()[:, 0]).cpu()
            if mode == 'train':
                max_frame = self.batch_size*3-1
            else:
                max_frame = self.batch_size - 1
            img = self.psf.forward(xyz_px, torch.squeeze(intensity).detach().cpu(), frame_ix, ix_low=int(frame_ix.min()), ix_high=max_frame)
            # print('frame_ix_min: ' + str(frame_ix.min()))
            # print('frame_ix_max: ' + str(frame_ix.max()))
            img = img.to(self.device)
        else:
            size = xnm.shape[0]
            xnm, ynm, Z, I = torch.reshape(xnm * self.pixel_size_x, (size,)), torch.reshape(ynm * self.pixel_size_y,
                                                                                            (size,)), \
                torch.reshape(Z, (size,)), torch.reshape(I, (size,))

            if self.robust_training:
                self.zernike[:, 2] = self.zernike_init[:, 2] + np.random.normal(loc=0, scale=self.vector_params.wavelength/100, size=21)
                self.VectorPSF = VectorPSFTorch(self.vector_params, self.zernike, self.objstage0)
            img = self.VectorPSF.simulate_parallel(xnm, ynm, Z, I)

            S = torch.reshape(S, (-1, self.train_size_x, self.train_size_y))
            img = place_psfs(self.vector_params, img, S, self.ph_scale)

        imgs_sim = img.reshape([-1, 1, self.train_size_x, self.train_size_y])
        imgs_sim = self.sim_noise(imgs_sim)

        return imgs_sim

    def simulate_image_eval(self, s_mask_all=None, xyzi_gt_all=None, S=None, xnm=None, ynm=None, Z=None, I=None):
        if self.psf_model == 'spline':
            molecule_tuple = tuple(s_mask_all.nonzero().transpose(1, 0))
            xyz_px = xyzi_gt_all[molecule_tuple[0], molecule_tuple[1], :3][:, [1, 0, 2]]
            xyz_px[:, 2] = xyz_px[:, 2] * self.z_scale
            xyz_px = xyz_px.cpu()
            intensity = (xyzi_gt_all[molecule_tuple[0], molecule_tuple[1], 3] * self.ph_scale).cpu()
            frame_ix = torch.squeeze(s_mask_all.nonzero()[:, 0]).cpu()

            img = self.psf.forward(xyz_px, torch.squeeze(intensity).detach().cpu(), frame_ix, ix_low=int(frame_ix.min()), ix_high=int(frame_ix.max()))
            # print('frame_ix_min: ' + str(frame_ix.min()))
            # print('frame_ix_max: ' + str(frame_ix.max()))
            img = img.to(self.device)
        else:
            size = xnm.shape[0]
            xnm, ynm, Z, I = torch.reshape(xnm * self.pixel_size_x, (size,)), torch.reshape(ynm * self.pixel_size_y,
                                                                                            (size,)), \
                torch.reshape(Z, (size,)), torch.reshape(I, (size,))

            img = self.VectorPSF.simulate_parallel(xnm, ynm, Z, I)

            S = torch.reshape(S, (-1, self.train_size_x, self.train_size_y))
            img = place_psfs(self.vector_params, img, S, self.ph_scale)

        imgs_sim = img.reshape([-1, 1, self.train_size_x, self.train_size_y])
        imgs_sim = self.sim_noise(imgs_sim)

        return imgs_sim

    def simulate_image_visualize(self, s_mask_all=None, xyzi_gt_all=None, S=None, xnm=None, ynm=None, Z=None, I=None):
        if self.psf_model == 'spline':
            molecule_tuple = tuple(s_mask_all.nonzero().transpose(1, 0))
            xyz_px = xyzi_gt_all[molecule_tuple[0], molecule_tuple[1], :3][:, [1, 0, 2]]
            xyz_px[:, 2] = xyz_px[:, 2] * self.z_scale
            xyz_px = xyz_px.cpu()
            intensity = (xyzi_gt_all[molecule_tuple[0], molecule_tuple[1], 3] * self.ph_scale).cpu()
            frame_ix = torch.squeeze(s_mask_all.nonzero()[:, 0]).cpu()

            img = self.psf.forward(xyz_px, torch.squeeze(intensity).detach().cpu(), frame_ix, ix_low=int(frame_ix.min()), ix_high=int(frame_ix.max()))
            # print('frame_ix_min: ' + str(frame_ix.min()))
            # print('frame_ix_max: ' + str(frame_ix.max()))
            img = img.to(self.device)
        else:
            size = xnm.shape[0]
            xnm, ynm, Z, I = torch.reshape(xnm * self.pixel_size_x, (size,)), torch.reshape(ynm * self.pixel_size_y,
                                                                                            (size,)), \
                torch.reshape(Z, (size,)), torch.reshape(I, (size,))

            img = self.VectorPSF.simulate_parallel(xnm, ynm, Z, I)

            S = torch.reshape(S, (-1, self.train_size_x, self.train_size_y))
            img = place_psfs(self.vector_params, img, S, self.ph_scale)

        imgs_sim = img.reshape([-1, 1, self.train_size_x, self.train_size_y])
        imgs_sim = self.sim_noise(imgs_sim)

        return imgs_sim

    def simulate_image_liteloc_addbgloss(self, s_mask_all=None, xyzi_gt_all=None, S=None, xnm=None, ynm=None, Z=None, I=None):
        if self.psf_model == 'spline':
            molecule_tuple = tuple(s_mask_all.nonzero().transpose(1, 0))
            xyz_px = xyzi_gt_all[molecule_tuple[0], molecule_tuple[1], :3][:, [1, 0, 2]]
            xyz_px[:, 2] = xyz_px[:, 2] * self.z_scale
            xyz_px = xyz_px.cpu()
            intensity = (xyzi_gt_all[molecule_tuple[0], molecule_tuple[1], 3] * self.ph_scale).cpu()
            frame_ix = torch.squeeze(s_mask_all.nonzero()[:, 0]).cpu()

            img = self.psf.forward(xyz_px, torch.squeeze(intensity).detach().cpu(), frame_ix, ix_low=int(frame_ix.min()), ix_high=self.batch_size*3-1)
            # print('frame_ix_min: ' + str(frame_ix.min()))
            # print('frame_ix_max: ' + str(frame_ix.max()))
            img = img.to(self.device)
        else:
            size = xnm.shape[0]
            xnm, ynm, Z, I = torch.reshape(xnm * self.pixel_size_x, (size,)), torch.reshape(ynm * self.pixel_size_y,
                                                                                            (size,)), \
                torch.reshape(Z, (size,)), torch.reshape(I, (size,))

            img = self.VectorPSF.simulate_parallel(xnm, ynm, Z, I)

            S = torch.reshape(S, (-1, self.train_size_x, self.train_size_y))
            img = place_psfs(self.vector_params, img, S, self.ph_scale)

        imgs_sim = img.reshape([-1, 1, self.train_size_x, self.train_size_y])
        psf_imgs_gt = imgs_sim.clone() / self.ph_scale * 10
        psf_imgs_gt = psf_imgs_gt.reshape([-1, 3, self.train_size_x, self.train_size_y])[:, 1]
        imgs_sim = self.sim_noise(imgs_sim)

        return imgs_sim, psf_imgs_gt

    def simulatedImg_torch(self, S, xnm, ynm, Z, I):

        size = xnm.shape[0]
        xnm, ynm, Z, I = torch.reshape(xnm * self.pixel_size_x, (size, )), torch.reshape(ynm * self.pixel_size_y, (size, )), \
                         torch.reshape(Z, (size, )), torch.reshape(I, (size, ))

        img = self.VectorPSF.simulate_parallel(xnm, ynm, Z, I)  #photons: (0.067, 1)

        S = torch.reshape(S, (-1, self.train_size_x, self.train_size_y))
        img_sim = place_psfs(self.vector_params, img, S, self.ph_scale)

        imgs_sim = img_sim.reshape([-1, 1, self.train_size_x, self.train_size_y])

        imgs_sim = self.sim_noise(imgs_sim)

        return imgs_sim

    def simulate_image_decode(self, s_mask_all=None, xyzi_gt_all=None, S=None, xnm=None, ynm=None, Z=None, I=None):
        if self.psf_model == 'spline':
            molecule_tuple = tuple(s_mask_all.nonzero().transpose(1, 0))
            xyz_px = xyzi_gt_all[molecule_tuple[0], molecule_tuple[1], :3][:, [1, 0, 2]]
            xyz_px[:, 2] = xyz_px[:, 2] * self.z_scale
            xyz_px = xyz_px.cpu()
            intensity = (xyzi_gt_all[molecule_tuple[0], molecule_tuple[1], 3] * self.ph_scale).cpu()
            frame_ix = torch.squeeze(s_mask_all.nonzero()[:, 0]).cpu()

            img = self.psf.forward(xyz_px, torch.squeeze(intensity).detach().cpu(), frame_ix, ix_low=int(frame_ix.min()), ix_high=self.batch_size*3-1)#int(frame_ix.max()))
            # print('frame_ix_min: ' + str(frame_ix.min()))
            # print('frame_ix_max: ' + str(frame_ix.max()))
            img = img.to(self.device)
        else:
            size = xnm.shape[0]
            xnm, ynm, Z, I = torch.reshape(xnm * self.pixel_size_x, (size,)), torch.reshape(ynm * self.pixel_size_y,
                                                                                            (size,)), \
                torch.reshape(Z, (size,)), torch.reshape(I, (size,))

            img = self.VectorPSF.simulate_parallel(xnm, ynm, Z, I)

            S = torch.reshape(S, (-1, self.train_size_x, self.train_size_y))
            img = place_psfs(self.vector_params, img, S, self.ph_scale)

        imgs_sim = img.reshape([-1, 1, self.train_size_x, self.train_size_y])
        psf_imgs_gt = imgs_sim.clone() / self.ph_scale * 10
        psf_imgs_gt = psf_imgs_gt.reshape([-1, 3, self.train_size_x, self.train_size_y])[:, 1]

        imgs_sim = self.sim_noise(imgs_sim)

        return imgs_sim, psf_imgs_gt

    def simulated_splinePSF_from_gt(self, xyz_px, intensity, frame_ix):

        img = self.psf.forward(xyz_px, intensity.detach().cpu(), frame_ix, ix_low=int(frame_ix.min()), ix_high=int(frame_ix.max()))
        imgs_sim = img.reshape([-1, 1, self.train_size_x, self.train_size_y]).to(self.device)
        imgs_sim = self.sim_noise(imgs_sim)

        return imgs_sim

    def generate_batch(self, size, local_context=False):

        M = np.ones([1, self.train_size_y, self.train_size_x])
        M = M / M.sum() * self.num_particles

        blink_p = torch.from_numpy(M).float().to(self.device) # torch.cuda.FloatTensor(M)
        blink_p = blink_p.reshape(1, 1, blink_p.shape[-2], blink_p.shape[-1]).repeat_interleave(size, 0)

        while True:
            locs = torch.distributions.Binomial(1, blink_p).sample().to(self.device)
            u = 0
            for i in range(self.batch_size):
                if locs[i].sum():
                    u = u + 1
            if u == self.batch_size:
                break

        zeros = torch.zeros_like(locs).to(self.device)

        # z position follows a uniform distribution with predefined range
        z = torch.distributions.Uniform(zeros - 1,
                                        zeros + 1).sample().to(self.device)

        # xy offset follow uniform distribution
        x_os = torch.distributions.Uniform(zeros - 0.5, zeros + 0.5).sample().to(self.device)
        y_os = torch.distributions.Uniform(zeros - 0.5, zeros + 0.5).sample().to(self.device)

        if local_context:
            surv_p = self.camera_params['surv_p']
            a11 = 1 - (1 - blink_p) * (1 - surv_p)
            locs2 = torch.distributions.Binomial(1, (1 - locs) * blink_p + locs * a11).sample().to(self.device)
            locs3 = torch.distributions.Binomial(1, (1 - locs2) * blink_p + locs2 * a11).sample().to(self.device)
            locs = torch.cat([locs, locs2, locs3], 1)
            x_os = x_os.repeat_interleave(3, 1)  # 直接复制 == 连续三帧的偏移量相同，但坐标不同 --> 全局坐标不同
            y_os = y_os.repeat_interleave(3, 1)
            z = z.repeat_interleave(3, 1)

        ints = torch.distributions.Uniform(torch.zeros_like(locs) + self.min_ph,
                                           torch.ones_like(locs)).sample().to(self.device)
        z *= locs
        x_os *= locs
        y_os *= locs
        ints *= locs

        xyzit = torch.cat([x_os[:, :, None], y_os[:, :, None], z[:, :, None], ints[:, :, None]], 2)
        xyzi = torch.cat([x_os.reshape([-1, 1, self.train_size_x, self.train_size_y]),
                          y_os.reshape([-1, 1, self.train_size_x, self.train_size_y]),
                          z.reshape([-1, 1, self.train_size_x, self.train_size_y]),
                          ints.reshape([-1, 1, self.train_size_x, self.train_size_y])], 1)

        X_os, Y_os, Z, I = self.transform_offsets(self.z_scale, locs.reshape([-1, self.train_size_x, self.train_size_y]), xyzi)
        xyzi_gt = torch.zeros([size, 0, 4]).float().to(self.device) #.type(torch.cuda.FloatTensor)
        s_mask = torch.zeros([size, 0]).float().to(self.device)

        xyzit = xyzit[:, 1] if local_context else xyzit[:, 0]

        # get all molecules' discrete pixel positions [number_in_batch, row, column]
        S = locs
        S = S[:, 1] if local_context else S[:, 0]
        s_inds = tuple(S.nonzero().transpose(1, 0))

        # get these molecules' sub-pixel xy offsets, z positions and photons
        xyzi_true = xyzit[s_inds[0], :, s_inds[1], s_inds[2]]

        # get the xy continuous pixel positions
        xyzi_true[:, 0] += s_inds[2].float().to(self.device) + 0.5
        xyzi_true[:, 1] += s_inds[1].float().to(self.device) + 0.5

        # return the gt numbers of molecules on each training images of this batch
        # (if local_context, return the number of molecules on the middle frame)
        s_counts = torch.unique_consecutive(s_inds[0], return_counts=True)[1]
        s_max = s_counts.max()

        # for each training images of this batch, build a molecule list with length=s_max
        xyzi_gt_curr = torch.zeros(size, s_max, 4).float().to(self.device) # torch.cuda.FloatTensor(size, s_max, 4).fill_(0)
        s_mask_curr = torch.zeros(size, s_max).float().to(self.device) # torch.cuda.FloatTensor(size, s_max).fill_(0)
        s_arr = torch.cat([torch.arange(c) for c in s_counts], dim=0)

        # put the gt in the molecule list, with remaining=0
        xyzi_gt_curr[s_inds[0], s_arr] = xyzi_true
        s_mask_curr[s_inds[0], s_arr] = 1

        xyzi_gt = torch.cat([xyzi_gt, xyzi_gt_curr], 1)
        s_mask = torch.cat([s_mask, s_mask_curr], 1)

        locs = locs.reshape([-1, self.train_size_x, self.train_size_y])
        return locs, X_os, Y_os, Z, I, s_mask, xyzi_gt, S

    def generate_batch_newest(self, size, local_context=False):
        # if val:
        #     M = np.ones([1, self.train_size_y, self.train_size_x])
        #     M[0, int(self.camera_params.margin_empty * self.train_size_y):int((1-self.camera_params.margin_empty) * self.train_size_y), int(self.camera_params.margin_empty * self.train_size_x):int((1-self.camera_params.margin_empty) * self.train_size_x)] += 9
        # else:
        #     M = np.zeros([1, self.train_size_y, self.train_size_x])
        #     M[0, int(self.camera_params.margin_empty * self.train_size_y):int((1-self.camera_params.margin_empty) * self.train_size_y),
        #     int(self.camera_params.margin_empty * self.train_size_x):int((1-self.camera_params.margin_empty) * self.train_size_x)] += 1
        # M = M / M.sum() * self.num_particles

        M = np.ones([1, self.train_size_y, self.train_size_x])
        M = M / M.sum() * self.num_particles

        blink_p = torch.from_numpy(M).float().to(self.device) # torch.cuda.FloatTensor(M)
        blink_p = blink_p.reshape(1, 1, blink_p.shape[-2], blink_p.shape[-1]).repeat_interleave(size, 0)

        while True:
            locs = torch.distributions.Binomial(1, blink_p).sample().to(self.device)
            u = 0
            for i in range(size):
                if locs[i].sum():
                    u = u + 1
            if u == size:
                break

        zeros = torch.zeros_like(locs).to(self.device)

        # z position follows a uniform distribution with predefined range
        z = torch.distributions.Uniform(zeros - 1,
                                        zeros + 1).sample().to(self.device)

        # xy offset follow uniform distribution
        x_os = torch.distributions.Uniform(zeros - 0.5, zeros + 0.5).sample().to(self.device)
        y_os = torch.distributions.Uniform(zeros - 0.5, zeros + 0.5).sample().to(self.device)

        if local_context:
            size = size * 3
            surv_p = self.camera_params.surv_p
            a11 = 1 - (1 - blink_p) * (1 - surv_p)
            locs2 = torch.distributions.Binomial(1, (1 - locs) * blink_p + locs * a11).sample().to(self.device)
            locs3 = torch.distributions.Binomial(1, (1 - locs2) * blink_p + locs2 * a11).sample().to(self.device)
            locs = torch.cat([locs, locs2, locs3], 1)
            x_os = x_os.repeat_interleave(3, 1)  # 直接复制 == 连续三帧的偏移量相同，但坐标不同 --> 全局坐标不同
            y_os = y_os.repeat_interleave(3, 1)
            z = z.repeat_interleave(3, 1)

        ints = torch.distributions.Uniform(torch.zeros_like(locs) + self.min_ph,
                                           torch.ones_like(locs)).sample().to(self.device)
        z *= locs
        x_os *= locs
        y_os *= locs
        ints *= locs

        # xyzit = torch.cat([x_os[:, :, None], y_os[:, :, None], z[:, :, None], ints[:, :, None]], 2)
        xyzi = torch.cat([x_os.reshape([-1, 1, self.train_size_x, self.train_size_y]),
                          y_os.reshape([-1, 1, self.train_size_x, self.train_size_y]),
                          z.reshape([-1, 1, self.train_size_x, self.train_size_y]),
                          ints.reshape([-1, 1, self.train_size_x, self.train_size_y])], 1)

        X_os, Y_os, Z, I = self.transform_offsets(self.z_scale, locs.reshape([-1, self.train_size_x, self.train_size_y]), xyzi)
        xyzi_gt = torch.zeros([size, 0, 4]).float().to(self.device)
        s_mask = torch.zeros([size, 0]).float().to(self.device)

        # xyzit_all = xyzit
        # xyzit = xyzit[:, 1] if local_context and spline_model else xyzit[:, 0]

        # get all molecules' discrete pixel positions [number_in_batch, row, column]
        s_inds = tuple(locs.reshape([-1, self.train_size_x, self.train_size_y]).nonzero().transpose(1, 0))

        # get these molecules' sub-pixel xy offsets, z positions and photons
        xyzi_true = xyzi[s_inds[0], :, s_inds[1], s_inds[2]]

        # get the xy continuous pixel positions
        xyzi_true[:, 0] += s_inds[2].float().to(self.device) + 0.5
        xyzi_true[:, 1] += s_inds[1].float().to(self.device) + 0.5

        # return the gt numbers of molecules on each training images of this batch
        # (if local_context, return the number of molecules on the middle frame)
        s_counts = torch.unique_consecutive(s_inds[0], return_counts=True)[1]
        s_max = s_counts.max()

        # for each training images of this batch, build a molecule list with length=s_max
        xyzi_gt_curr = torch.zeros(size, s_max, 4).float().to(self.device) # torch.cuda.FloatTensor(size, s_max, 4).fill_(0)
        s_mask_curr = torch.zeros(size, s_max).float().to(self.device) # torch.cuda.FloatTensor(size, s_max).fill_(0)
        s_arr = torch.cat([torch.arange(c) for c in s_counts], dim=0)

        # put the gt in the molecule list, with remaining=0
        xyzi_gt_curr[s_inds[0], s_arr] = xyzi_true
        s_mask_curr[s_inds[0], s_arr] = 1

        xyzi_gt = torch.cat([xyzi_gt, xyzi_gt_curr], 1)
        s_mask = torch.cat([s_mask, s_mask_curr], 1)

        locs = locs.reshape([-1, self.train_size_x, self.train_size_y])
        return locs, X_os, Y_os, Z, I, s_mask, xyzi_gt

    def generate_batch_deepstorm3d(self, size, val):
        # if we're testing then seed the random generator
        # if self.seed is not None:
        #     np.random.seed(self.seed)
        # randomly vary the number of emitters
        #fy
        M = np.ones([1, self.train_size_y, self.train_size_x])
        M = M / M.sum() * self.num_particles

        blink_p = torch.from_numpy(M).float().to(self.device)
        blink_p = blink_p.reshape(1, 1, blink_p.shape[-2], blink_p.shape[-1]).repeat_interleave(size, 0)
        while True:
            locs = torch.distributions.Binomial(1, blink_p).sample().to(self.device)
            u = 0
            for i in range(self.batch_size):
                if locs[i].sum():
                    u = u + 1
            if u == self.batch_size:
                break

        zeros = torch.zeros_like(locs).to(self.device)
        # z position follows a uniform distribution with predefined range
        z = torch.distributions.Uniform(zeros - 1,
                                        zeros + 1).sample().to(self.device)
        # xy offset follow uniform distribution
        x_os = torch.distributions.Uniform(zeros - 0.5, zeros + 0.5).sample().to(self.device)
        y_os = torch.distributions.Uniform(zeros - 0.5, zeros + 0.5).sample().to(self.device)

        ints = torch.distributions.Uniform(torch.zeros_like(locs) + self.min_ph,
                                           torch.ones_like(locs)).sample().to(self.device)
        z *= locs

        x_os *= locs
        y_os *= locs

        ints *= locs
        S = torch.squeeze(locs)
        xyzit = torch.cat([x_os[:, :, None], y_os[:, :, None], z[:, :, None], ints[:, :, None]], 2)
        xyzi = torch.cat([x_os.reshape([-1, 1, self.train_size_x, self.train_size_y]),
                          y_os.reshape([-1, 1, self.train_size_x, self.train_size_y]),
                          z.reshape([-1, 1, self.train_size_x, self.train_size_y]),
                          ints.reshape([-1, 1, self.train_size_x, self.train_size_y])], 1)

        X_os, Y_os, Z, I = self.transform_offsets(self.z_scale, locs.reshape([-1, self.train_size_x, self.train_size_y]), xyzi)
        xyzi_gt = torch.zeros([size, 0, 4]).float().to(self.device)
        s_mask = torch.zeros([size, 0]).float().to(self.device)

        xyzit = xyzit[:, 0]
        # get all molecules' discrete pixel positions [number_in_batch, row, column]
        S = S.reshape([-1, self.train_size_x, self.train_size_y])
        s_inds = tuple(S.nonzero().transpose(1, 0))
        # get these molecules' sub-pixel xy offsets, z positions and photons
        xyzi_true = xyzit[s_inds[0], :, s_inds[1], s_inds[2]]
        # get the xy continuous pixel positions
        xyzi_true[:, 0] += s_inds[2].float().to(self.device) + 0.5
        xyzi_true[:, 1] += s_inds[1].float().to(self.device) + 0.5
        # return the gt numbers of molecules on each training images of this batch
        # (if local_context, return the number of molecules on the middle frame)
        s_counts = torch.unique_consecutive(s_inds[0], return_counts=True)[1]
        s_max = s_counts.max()
        # for each training images of this batch, build a molecule list with length=s_max
        xyzi_gt_curr = torch.zeros(size, s_max, 4).float().to(self.device) # torch.cuda.FloatTensor(size, s_max, 4).fill_(0)
        s_mask_curr = torch.zeros(size, s_max).float().to(self.device) # torch.cuda.FloatTensor(size, s_max).fill_(0)
        s_arr = torch.cat([torch.arange(c) for c in s_counts], dim=0)
        # put the gt in the molecule list, with remaining=0
        xyzi_gt_curr[s_inds[0], s_arr] = xyzi_true
        s_mask_curr[s_inds[0], s_arr] = 1

        xyzi_gt = torch.cat([xyzi_gt, xyzi_gt_curr], 1)
        s_mask = torch.cat([s_mask, s_mask_curr], 1)

        locs = locs.reshape([-1, self.train_size_x, self.train_size_y])
        return locs, X_os, Y_os, Z, I, s_mask, xyzi_gt, xyzi

    def sim_noise(self, imgs_sim, add_noise=True):
        if self.camera_params.camera == 'EMCCD':
            bg_photons = (self.bg - self.camera_params.baseline) / self.camera_params.em_gain \
                         * self.camera_params.e_per_adu / self.camera_params.qe
            if bg_photons < 0:
                print('converted bg_photons is less than 0, please check the parameters setting!')

            if self.perlin_noise:
                size_x, size_y = imgs_sim.shape[-2], imgs_sim.shape[-1]
                self.PN_octaves_num = 1
                space_range_x = size_x / self.pn_res
                space_range_y = size_y / self.pn_res
                self.PN = PerlinNoiseFactory(dimension=2, octaves=self.PN_octaves_num,
                                             tile=(space_range_x, space_range_y),
                                             unbias=True)
                PN_tmp_map = np.zeros([size_x, size_y])
                for x in range(size_x):
                    for y in range(size_y):
                        cal_PN_tmp = self.PN(x / self.pn_res, y / self.pn_res)
                        PN_tmp_map[x, y] = cal_PN_tmp
                PN_noise = PN_tmp_map * bg_photons * self.pn_factor
                bg_photons += PN_noise
                bg_photons = gpu_cpu_torch (bg_photons, self.device)

            imgs_sim += bg_photons

            if add_noise:
                imgs_sim = torch.distributions.Poisson(
                    imgs_sim * self.camera_params.qe + self.camera_params.spurious_c).sample()

                imgs_sim = torch.distributions.Gamma(imgs_sim, 1 / self.camera_params.em_gain).sample()

                RN = self.camera_params.sig_read
                zeros = torch.zeros_like(imgs_sim)
                read_out_noise = torch.distributions.Normal(zeros, zeros + RN).sample()

                imgs_sim = imgs_sim + read_out_noise
                imgs_sim = torch.clamp(imgs_sim / self.camera_params.e_per_adu + self.camera_params.baseline, min=0)

        elif self.camera_params.camera == 'sCMOS':
            bg_photons = (self.bg - self.camera_params.baseline) \
                         * self.camera_params.e_per_adu / self.camera_params.qe
            if bg_photons < 0:
                print('converted bg_photons is less than 0, please check the parameters setting!')

            if self.perlin_noise:
                size_x, size_y = imgs_sim.shape[-2], imgs_sim.shape[-1]
                self.PN_octaves_num = 1
                space_range_x = size_x / self.pn_res
                space_range_y = size_y / self.pn_res
                self.PN = PerlinNoiseFactory(dimension=2, octaves=self.PN_octaves_num,
                                             tile=(space_range_x, space_range_y),
                                             unbias=True)
                PN_tmp_map = np.zeros([size_x, size_y])
                for x in range(size_x):
                    for y in range(size_y):
                        cal_PN_tmp = self.PN(x / self.pn_res, y / self.pn_res)
                        PN_tmp_map[x, y] = cal_PN_tmp
                PN_noise = PN_tmp_map * bg_photons * self.pn_factor
                bg_photons += PN_noise
                bg_photons = gpu_cpu_torch(bg_photons, self.device)

            imgs_sim += bg_photons

            if add_noise:
                imgs_sim = torch.distributions.Poisson(
                    imgs_sim * self.camera_params.qe + self.camera_params.spurious_c).sample()

                RN = self.camera_params.sig_read
                zeros = torch.zeros_like(imgs_sim)
                read_out_noise = torch.distributions.Normal(zeros, zeros + RN).sample()

                imgs_sim = imgs_sim + read_out_noise
                imgs_sim = torch.clamp(imgs_sim / self.camera_params.e_per_adu + self.camera_params.baseline, min=0)
        else:
            print('wrong camera types! please choose EMCCD or sCMOS!')
        return imgs_sim

    def transform_offsets(self,z, S, XYZI):
        n_samples = S.shape[0] // XYZI.shape[0]
        XYZI_rep = XYZI.repeat_interleave(n_samples, 0)

        s_inds = tuple(S.nonzero().transpose(1, 0))
        x_os_vals = (XYZI_rep[:, 0][s_inds])[:, None, None]
        y_os_vals = (XYZI_rep[:, 1][s_inds])[:, None, None]
        z_vals = z * XYZI_rep[:, 2][s_inds][:, None, None]
        i_vals = (XYZI_rep[:, 3][s_inds])[:, None, None]

        return x_os_vals, y_os_vals, z_vals, i_vals

    def transform_grid(self, z_scale, S, xyzi_gt):

        interval_len = self.z_scale * 2 / 121  # 每个区间的距离，[nm]
        boolean_grid_batch = torch.Tensor(torch.Size([self.batch_size, 121, self.train_size_y * 4,
                                                      self.train_size_x * 4]))  # [[] for i in range(self.batch_size)]

        for i in range(self.batch_size):
            molecule_n = S[i].nonzero().shape[0]
            indX = np.array(np.floor((xyzi_gt[i, :molecule_n][:, 0] * 4).cpu())).astype('int').flatten(
                'F').tolist()  # exchange x and y
            indY = np.array(np.floor((xyzi_gt[i, :molecule_n][:, 1] * 4).cpu())).astype('int').flatten(
                'F').tolist()  # exchange x and y
            indZ = np.array(np.floor((z_scale * (xyzi_gt[i, :molecule_n][:, 2] + 1) / interval_len).cpu())).astype(
                'int').flatten('F').tolist()
            print(indZ)
            ibool = torch.LongTensor([indZ, indY, indX])
            vals = torch.ones(molecule_n)

            boolean_grid = torch.sparse.FloatTensor(ibool, vals, torch.Size(
                [121, self.train_size_y * 4, self.train_size_x * 4])).to_dense()

            boolean_grid_batch[i] = boolean_grid
        return boolean_grid_batch


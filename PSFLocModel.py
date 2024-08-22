import collections
import time
import pickle
import os
import thop
import numpy as np
import pandas as pd
import torch.cuda
import tqdm
import matplotlib.pyplot as plt
import tifffile as tif
import torch.nn.functional as F
from torch.optim import NAdam
from torch.cuda.amp import autocast
import itertools
from torch.utils.data import DataLoader

from utils.help_utils import calculate_bg, cpu, gpu, InferDataset
from network.loss_utils import LossFuncs
from utils.dataGenerator import DataGenerator
from network.liteloc import LiteLoc, LiteLoc_wo_local
from network.decode import DECODE
from network.eval_utils import EvalMetric
from PSF_vector_gpu.vectorpsf import VectorPSFTorch

class LitelocModel:
    def __init__(self, params):

        if params.Training.bg is None:
            params.Training.bg = calculate_bg(params.Training.infer_data)

        self.DataGen = DataGenerator(params.Training, params.Camera, params.PSF_model)

        self.LiteLoc = LiteLoc().to(torch.device('cuda'))

        self.EvalMetric = EvalMetric(params.PSF_model, params.Training)

        self.net_weight = list(self.LiteLoc.parameters())

        self.optimizer = NAdam(self.net_weight, lr=8e-4, betas=(0.8, 0.8888), eps=1e-8)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.85)

        self.criterion = LossFuncs(train_size=params.Training.train_size[0])

        self.DataGen.genValidData()
        self.valid_data = self.DataGen.readValidFile()

        self.start_epoch = 0

        self.recorder = {}
        self.init_recorder()

        self.no_improve = 0  # 7次无变化 终止
        self.best_loss = np.nan
        self.best_jaccard = np.nan

        self.params = params

    def init_recorder(self):

        self.recorder['cost_hist'] = collections.OrderedDict([])
        self.recorder['recall'] = collections.OrderedDict([])
        self.recorder['precision'] = collections.OrderedDict([])
        self.recorder['jaccard'] = collections.OrderedDict([])
        self.recorder['rmse_lat'] = collections.OrderedDict([])
        self.recorder['rmse_ax'] = collections.OrderedDict([])
        self.recorder['rmse_vol'] = collections.OrderedDict([])
        self.recorder['jor'] = collections.OrderedDict([])
        self.recorder['eff_lat'] = collections.OrderedDict([])
        self.recorder['eff_ax'] = collections.OrderedDict([])
        self.recorder['eff_3d'] = collections.OrderedDict([])
        self.recorder['update_time'] = collections.OrderedDict([])

    def train_spline(self):

        print(self.LiteLoc)
        print("number of parameters: ", sum(param.numel() for param in self.LiteLoc.parameters()))
        dummy_input = torch.randn(3, 1, 128, 128).cuda()
        macs, parameter = thop.profile(self.LiteLoc, inputs=(dummy_input,))
        macs, parameter = thop.clever_format([macs, parameter], '%.3f')
        print(f'Params:{parameter}, MACs:{macs}, (input shape:{dummy_input.shape})')

        print('start training!')

        while self.start_epoch < self.params.Training.max_epoch:
            if self.no_improve == 7:
                break

            tot_cost = []
            tt = time.time()
            local_context = True
            for i in range(0, self.params.Training.eval_iteration):

                locs, X, Y, Z, I, s_mask, xyzi_gt = self.DataGen.generate_batch_newest(self.params.Training.batch_size, val=False, local_context=local_context)

                imgs_sim = self.DataGen.simulate_image(s_mask, xyzi_gt, locs, X, Y, Z, I)

                if local_context:
                    imgs_sim = imgs_sim.reshape([self.params.Training.batch_size, 3, locs.shape[-2], locs.shape[-1]])
                    mid_frame = torch.arange(1, locs.shape[0], 3)
                    xyzi_gt = xyzi_gt[mid_frame]
                    s_mask = s_mask[mid_frame]
                    locs = locs[mid_frame]

                p, xyzi_est, xyzi_sig = self.LiteLoc.forward(imgs_sim, test=False)

                loss_total = self.criterion.final_loss(p, xyzi_est, xyzi_sig, xyzi_gt, s_mask, locs)

                self.optimizer.zero_grad()

                loss_total.backward()
                # avoid too large gradient
                torch.nn.utils.clip_grad_norm_(list(self.LiteLoc.parameters()), max_norm=0.03, norm_type=2)

                # update the network and the optimizer state
                self.optimizer.step()
                self.scheduler.step()

                tot_cost.append(cpu(loss_total))

            self.start_epoch += 1
            print(f"Epoch{self.start_epoch}/{self.params.Training.max_epoch}")
            self.recorder['cost_hist'][self.start_epoch] = np.mean(tot_cost)
            self.recorder['update_time'][self.start_epoch] = (time.time() - tt) * 1000 / self.params.Training.eval_iteration

            self.evaluation_spline()  # todo: consecutive three frames like inference;
            torch.cuda.empty_cache()
            self.save_model()

            print('{}{}{:0.3f}'.format(' || ', 'Cost: ', self.recorder['cost_hist'][self.start_epoch]), end='')
            print('{}{}{}'.format(' || ', 'BatchNr.: ', self.params.Training.eval_iteration * (self.start_epoch)), end='')
            print('{}{}{:0.1f}{}'.format(' || ', 'Time Upd.: ', self.recorder['update_time'][self.start_epoch], ' ms '))

        print('training finished!')

    def evaluation_spline(self):  # todo: generate three consecutive frames
        self.LiteLoc.eval()
        loss = 0
        pred_list = []
        truth_list = []

        with torch.set_grad_enabled(False):
            for batch_ind, (xemit, yemit, z, S, Nphotons, s_mask, gt) in enumerate(
                    self.valid_data):
                img_sim = self.DataGen.simulate_image(s_mask[0], gt[0], S, torch.squeeze(xemit),
                                                      torch.squeeze(yemit), torch.squeeze(z), torch.squeeze(Nphotons))

                P, xyzi_est, xyzi_sig = self.LiteLoc.forward(img_sim, test=True)
                gt, s_mask, S = gt[:, 1:-1], s_mask[:, 1:-1], S[:, 1:-1]
                loss_total = self.criterion.final_loss(P, xyzi_est, xyzi_sig, gt, s_mask, S)

                pred, match = self.EvalMetric.predlist(P, xyzi_est, gt, batch_ind)

                pred_list = pred_list + pred
                truth_list = truth_list + match

                loss = loss + loss_total

        pred_dict, match = self.EvalMetric.limited_matching(truth_list, pred_list)
        for k in self.recorder.keys():
            if k in pred_dict:
                self.recorder[k][self.start_epoch] = pred_dict[k]

    def save_model(self):
        if not (os.path.isdir(self.params.Training.result_path)):
            os.mkdir(self.params.Training.result_path)
        path_checkpoint = self.params.Training.result_path + 'checkpoint.pkl'
        torch.save(self, path_checkpoint)

    def calculate_crlb_rmse(self, zstack=25, sampling_num=100):  # for vector psf
        PSF_torch = VectorPSFTorch(psf_params=self.params.PSF_model.vector_psf, zernike_aber=self.DataGen.zernike_aber)
        xemit = torch.tensor(0 * np.ones(zstack))
        yemit = torch.tensor(0 * np.ones(zstack))
        zemit = torch.tensor(1 * np.linspace(-self.params.PSF_model.z_scale, self.params.PSF_model.z_scale, zstack))
        Nphotons = torch.tensor((self.params.Training.photon_range[0] + self.params.Training.photon_range[1]) / 2 * np.ones(zstack)).cuda()
        bg = torch.tensor(
            (self.params.Training.bg - self.params.Camera.baseline) / self.params.Camera.em_gain *
            self.params.Camera.e_per_adu / self.params.Camera.qe * np.ones(zstack)).cuda()

        # calculate crlb and plot
        crlb_xyz, _ = PSF_torch.compute_crlb(xemit, yemit, zemit, Nphotons, bg)
        plt.figure(constrained_layout=True)
        plt.plot(zemit, cpu(crlb_xyz[:, 0]),'b', zemit, cpu(crlb_xyz[:, 1]),'g', zemit, cpu(crlb_xyz[:, 2]),'r')
        plt.legend(('$CRLB_x^{1/2}$', '$CRLB_y^{1/2}$', '$CRLB_z^{1/2}$'), ncol=3, loc='upper center')
        plt.xlim([-self.params.PSF_model.z_scale, self.params.PSF_model.z_scale])
        plt.show()

        # simulate single-molecule data
        xemit = (np.ones(zstack) - 2 * np.random.rand(1)) * self.params.PSF_model.vector_psf.pixel_size_xy[0]
        yemit = (np.ones(zstack) - 2 * np.random.rand(1)) * self.params.PSF_model.vector_psf.pixel_size_xy[1]
        zemit = torch.tensor(1 * np.linspace(-self.params.PSF_model.z_scale, self.params.PSF_model.z_scale, zstack))
        sampling_data = [[] for i in range(sampling_num*zstack)]
        sampling_gt = [[] for i in range(sampling_num*zstack)]
        frame_count = 0
        for i in tqdm.tqdm(range(sampling_num)):
            ground_truth = [[] for k in range(zstack)]
            for j in range(zstack):
                frame_count = frame_count + 1
                ground_truth[j] = [frame_count,
                                   xemit[j] + self.params.PSF_model.vector_psf.Npixels / 2 * self.params.PSF_model.vector_psf.pixel_size_xy[0] +
                                   self.params.PSF_model.vector_psf.pixel_size_xy[0],
                                   yemit[j] + self.params.PSF_model.vector_psf.Npixels / 2 * self.params.PSF_model.vector_psf.pixel_size_xy[1] +
                                   self.params.PSF_model.vector_psf.pixel_size_xy[1],
                                   zemit[j] + 0, cpu(Nphotons[j])]
            psfs = PSF_torch.simulate_parallel(gpu(xemit), gpu(yemit), zemit.cuda(), Nphotons)  # xyz's reference is center of image
            psfs = F.pad(psfs, pad=(1, 0, 1, 0), mode='constant', value=0)
            data = psfs + bg[:, None, None]

            # sampling_data[i*zstack:(i+1)*zstack] = self.DataGen.sim_noise(torch.unsqueeze(psfs, dim=1))
            sampling_data[i * zstack:(i + 1) * zstack] = torch.tensor(np.random.poisson(cpu(data))).unsqueeze(dim=1)
            sampling_gt[i*zstack:(i+1)*zstack] = ground_truth



        sampling_data = torch.cat(sampling_data, dim=0).to(torch.float32).cuda()

        '''image_path = "/home/feiyue/LiteLoc_local_torchsimu/CRLB_sampling_data_0815_parallel/Astigmatism_single_molecule.tif"
        gt_path = "/home/feiyue/LiteLoc_local_torchsimu/CRLB_sampling_data_0815_parallel/Astigmatism_single_molecule_nonNo.csv"
        sampling_gt = np.array(pd.read_csv(gt_path)).tolist()
        sampling_data = gpu(tif.imread(image_path))'''

        liteloc_pred_list = torch.zeros([10000000, 6]).cuda()
        liteloc_index_0 = 0
        with torch.no_grad():
            with autocast():
                for i in range(int(np.ceil(sampling_num*zstack/self.params.Training.batch_size))):
                    img = sampling_data[i*self.params.Training.batch_size:(i+1)*self.params.Training.batch_size]
                    liteloc_molecule_tensor = self.LiteLoc.analyze(img, threshold=0.98)
                    liteloc_molecule_tensor[:, 0] += i * self.params.Training.batch_size
                    liteloc_molecule_tensor[:, 1] = liteloc_molecule_tensor[:, 1] * \
                                                    self.params.PSF_model.vector_psf.pixel_size_xy[0]
                    liteloc_molecule_tensor[:, 2] = liteloc_molecule_tensor[:, 2] * \
                                                    self.params.PSF_model.vector_psf.pixel_size_xy[1]
                    liteloc_molecule_tensor[:, 3] = liteloc_molecule_tensor[:, 3] * self.params.PSF_model.z_scale
                    liteloc_molecule_tensor[:, 4] = liteloc_molecule_tensor[:, 4] * self.params.Training.photon_range[1]
                    liteloc_pred_list[
                    liteloc_index_0:liteloc_index_0 + liteloc_molecule_tensor.shape[0]] = liteloc_molecule_tensor
                    liteloc_index_0 = liteloc_molecule_tensor.shape[0] + liteloc_index_0

                liteloc_pred = cpu(liteloc_pred_list[:liteloc_index_0]).tolist()
        liteloc_perf_dict, liteloc_matches = self.EvalMetric.limited_matching(sampling_gt, liteloc_pred)

        dz = np.abs(zemit[2] - zemit[1])
        matches = torch.tensor(liteloc_matches)
        rmse_xyz = np.zeros([3, zstack])
        for i in range(zstack):
            z = zemit[i]
            ind = np.where(((z - dz / 2) < matches[:, 2]) & (matches[:, 2] < (z + dz / 2)))
            tmp = np.squeeze(matches[ind, :])
            if tmp.dim() == 1:
                tmp = torch.unsqueeze(tmp, dim=0)
            rmse_xyz[0, i] = np.sqrt(torch.mean(np.square(tmp[:, 0] - tmp[:, 4])))
            rmse_xyz[1, i] = np.sqrt(torch.mean(np.square(tmp[:, 1] - tmp[:, 5])))
            rmse_xyz[2, i] = np.sqrt(torch.mean(np.square(tmp[:, 2] - tmp[:, 6])))

        plt.figure(constrained_layout=True, dpi=500.0)
        plt.rcParams['axes.facecolor'] = 'white'
        plt.plot(zemit, cpu(crlb_xyz)[:, 0], '#1f77b4', zemit, cpu(crlb_xyz)[:, 1], '#2ca02c',
                 zemit, cpu(crlb_xyz)[:, 2], '#ff7f0e')
        plt.scatter(zemit, rmse_xyz[0, :], c='#1f77b4', marker='o')
        plt.scatter(zemit, rmse_xyz[1, :], c='#2ca02c', marker='o')
        plt.scatter(zemit, rmse_xyz[2, :], c='#ff7f0e', marker='o')
        labelss = plt.legend(('$CRLB_x^{1/2}$', '$CRLB_y^{1/2}$', '$CRLB_z^{1/2}$', '$LiteLoc\ RMSE_x$',
                              '$LiteLoc\ RMSE_y$', '$LiteLoc\ RMSE_z$'), ncol=2,
                             loc='upper center').get_texts()
        plt.xlim([-self.params.PSF_model.z_scale, self.params.PSF_model.z_scale])
        x_ticks = np.arange(-self.params.PSF_model.z_scale, self.params.PSF_model.z_scale+1, 200)
        plt.xticks(x_ticks)
        plt.ylim(bottom=0)
        plt.tick_params(labelsize=14)
        plt.show()


class DECODEModel:
    def __init__(self, net_params, camera_params, psf_params, train_params, eval_params):

        self.DataGen = DataGenerator(train_params, camera_params, psf_params)

        self.DECODE = DECODE(net_params).to(torch.device('cuda'))

        self.EvalM = Eval(eval_params, train_params)

        self.net_weights = list(self.DECODE.frame_module.parameters()) + list(self.DECODE.context_module.parameters()) + list(self.DECODE.out_module.parameters())

        self.optimizer = torch.optim.AdamW(self.net_weights, lr=net_params['lr'], weight_decay=net_params['w_decay'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=net_params['lr_decay'])

        self.criterion = LossFuncs_decode(train_size=train_params['train_size'][0])

        if train_params['dataresume']:
            self.validation_generator, self.validNum = self.DataGen.readValidFile()
        else:
            self.validation_generator, self.validNum = self.DataGen.genValidData()

        self.start_epoch = 0
        self.endEpoch = train_params['max_iters']
        self.pathSave = train_params['results_path']
        self.interval = train_params['interval']
        self.batch_size = train_params['batch_size']
        self.ph_scale = psf_params['ph_scale']

        if train_params['netresume']:
            self.loadConfig()

        self.recorder = {}
        self.init_recorder()

        self.no_improve = 0  # 7次无变化 终止
        self.best_loss = np.nan
        self.best_jaccard = np.nan

    def init_recorder(self):

        self.recorder['cost_hist'] = collections.OrderedDict([])
        self.recorder['recall'] = collections.OrderedDict([])
        self.recorder['precision'] = collections.OrderedDict([])
        self.recorder['jaccard'] = collections.OrderedDict([])
        self.recorder['rmse_lat'] = collections.OrderedDict([])
        self.recorder['rmse_ax'] = collections.OrderedDict([])
        self.recorder['rmse_vol'] = collections.OrderedDict([])
        self.recorder['jor'] = collections.OrderedDict([])
        self.recorder['eff_lat'] = collections.OrderedDict([])
        self.recorder['eff_ax'] = collections.OrderedDict([])
        self.recorder['eff_3d'] = collections.OrderedDict([])
        self.recorder['update_time'] = collections.OrderedDict([])

    def train(self, printmodel=True):

        if printmodel:
            print('DECODE architecture')
            print('=' * 50)
            print(self.DECODE)
            print("number of parameters: ", sum(param.numel() for param in self.DECODE.parameters()))

            dummy_input = torch.randn(1, 1, 128, 128).cuda()
            macs, params = thop.profile(self.DECODE, inputs=(dummy_input,))
            macs, params = thop.clever_format([macs, params], '%.3f')
            print(f'Params:{params}, MACs:{macs}, (input shape:{dummy_input.shape})')

        print('start training!')
        while self.start_epoch < self.endEpoch:
            if self.no_improve == 7:
                break
            tot_cost = []
            tt = time.time()
            for i in range(0, self.interval):  # todo: add S variable
                locs, X, Y, Z, I, s_mask, xyzi_gt, S = self.DataGen.generate_batch(self.batch_size, val=False, local_context=False)  # todo: simulated extra two or three frames
                imgs_sim = self.DataGen.simulatedImg_torch_decode(locs, X, Y, Z, I)

                imgs_sim = imgs_sim.reshape([-1, 1, locs.shape[-2], locs.shape[-1]])

                #p, xyzi_est, xyzi_sig, psf_imgs_est = self.DECODE.forward(imgs_sim)
                p, xyzi_est, xyzi_sig = self.DECODE.forward(imgs_sim)

                loss_total = self.criterion.final_loss(p, xyzi_est, xyzi_sig, xyzi_gt, s_mask) #, psf_imgs_est, psf_imgs_gt)

                self.optimizer.zero_grad()

                loss_total.backward()
                # avoid too large gradient
                torch.nn.utils.clip_grad_norm_(list(self.DECODE.parameters()), max_norm=0.03, norm_type=2)

                # update the network and the optimizer state
                self.optimizer.step()
                self.scheduler.step()

                tot_cost.append(cpu(loss_total))

            self.start_epoch += 1
            print(f"Epoch{self.start_epoch}/{self.endEpoch}")
            self.recorder['cost_hist'][self.start_epoch] = np.mean(tot_cost)
            self.recorder['update_time'][self.start_epoch] = (time.time() - tt) * 1000 / self.interval

            self.evaluation()  # todo: consistent with 'test' process; generate data for evaluation only once
            self.saveStatus()
            self.printResult()
        print('training finished!')

    def evaluation(self):  # todo: use the new post-process program
        # enable interactive plotting throughout iterations

        self.DECODE.eval()
        loss = 0
        pred_list = []
        truth_list = []
        with torch.set_grad_enabled(False):
            for batch_ind, (xemit, yemit, z, S, Nphotons, s_mask, gt) in enumerate(
                    self.validation_generator):
                xemit, y_emit, z, Nphotons = torch.squeeze(xemit), torch.squeeze(yemit), torch.squeeze(
                    z), torch.squeeze(Nphotons)
                imgs_sim = self.DataGen.simulatedImg_torch_decode(S, xemit, yemit, z, Nphotons)

                P, xyzi_est, xyzi_sig = self.DECODE.forward(imgs_sim)
                loss_total = self.criterion.final_loss(P, xyzi_est, xyzi_sig, gt, s_mask) #, psf_imgs_est, psf_imgs_gt)

                pred, match = self.EvalM.predlist(P, xyzi_est, gt, batch_ind)

                pred_list = pred_list + pred
                truth_list = truth_list + match

                loss = loss + loss_total

        pred_dict, match = self.EvalM.limited_matching(truth_list,pred_list)
        for k in self.recorder.keys():
            if k in pred_dict:
                self.recorder[k][self.start_epoch] = pred_dict[k]

        self.earlyStopping(loss/self.validNum, pred_dict['jaccard'])

    def loadConfig(self):
        print("=> loading checkpoint to resume training")
        checkpoint = torch.load(self.pathSave + "best_checkpoint_1.pth.tar")
        self.start_epoch = checkpoint['epoch']
        self.DECODE.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.last_epoch = self.start_epoch
        print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))

    def saveStatus(self,name=None):
        if not (os.path.isdir(self.pathSave)):
            os.mkdir(self.pathSave)
        checkpoint = {"state_dict": self.DECODE.state_dict(),
                      "optimizer": self.optimizer.state_dict(),
                      "epoch": self.start_epoch}
        if name:
            path_checkpoint = self.pathSave + name+'checkpoint.pth.tar'
            path_result = self.pathSave + name+'result.pkl'

        else:
            path_checkpoint = self.pathSave + 'checkpoint.pth.tar'
            path_result = self.pathSave + 'result.pkl'
        torch.save(checkpoint, path_checkpoint)
        with open(path_result, 'wb') as f:
            pickle.dump(self.recorder, f)

    def printResult(self):
        print('{}{}{:0.3f}'.format(' || ', 'Cost: ', self.recorder['cost_hist'][self.start_epoch]), end='')
        print('{}{}{}'.format(' || ', 'BatchNr.: ', self.interval*(self.start_epoch)), end='')
        print('{}{}{:0.1f}{}'.format(' || ', 'Time Upd.: ', self.recorder['update_time'][self.start_epoch], ' ms '))

    def earlyStopping(self,loss,jaccard):
        if self.best_loss == np.nan or self.best_loss > loss:
            self.best_loss = loss
            self.best_jaccard = jaccard
            self.saveStatus("best")
            return

        if self.best_loss <= loss and self.best_jaccard >= jaccard:
            self.no_improve += 1
            print(f"No Improve for{self.no_improve} epoch!")
            return

        if self.best_jaccard < jaccard:
            self.best_loss = loss
            self.best_jaccard = jaccard
            self.saveStatus("best")





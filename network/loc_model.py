import collections
import time
import os

import thop
import numpy as np
import torch.cuda
import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.optim import NAdam
from torch.cuda.amp import autocast

from utils.help_utils import calculate_bg, cpu, gpu, save_yaml, create_infer_yaml
from network.loss_utils import LossFuncs
from utils.data_generator import DataGenerator
from network.liteloc import LiteLoc
from utils.eval_utils import EvalMetric
from vector_psf.vectorpsf import VectorPSFTorch


class LocModel:
    def __init__(self, params, device = 'cuda'):

        if device == 'cuda':
            assert torch.cuda.is_available()
            torch.backends.cudnn.benchmark = True

        self.network = LiteLoc().to(device)
        self.network.get_parameter_number()

        if params.Training.infer_data is not None:
            params.Training.bg, params.Training.photon_range = calculate_bg(params)

        real_bg = (params.Training.bg - params.Camera.baseline) / params.Camera.em_gain * params.Camera.e_per_adu / params.Camera.qe

        print('image background is: ' + str(params.Training.bg))
        print('real background (with camera model) is: ' + str(real_bg))

        print('signal photon range is: (' + str(params.Training.photon_range[0]) +', ' + str(params.Training.photon_range[1]) + ')')

        self.DataGen = DataGenerator(params.Training, params.Camera, params.PSF_model, device)

        self.EvalMetric = EvalMetric(params.PSF_model, params.Training)

        self.net_weight = list(self.network.parameters())

        self.optimizer = NAdam(self.net_weight, lr=8e-4, betas=(0.8, 0.8888), eps=1e-8)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.85)

        if params.Training.model_init is not None:
            checkpoint = torch.load(params.Training.model_init, map_location=device)
            self.start_epoch = checkpoint.start_epoch
            print('continue to train from epoch ' + str(self.start_epoch))
            self.network.load_state_dict(checkpoint.LiteLoc.state_dict(), strict=False)
            self.optimizer.load_state_dict(checkpoint.optimizer.state_dict())
            self.scheduler.last_epoch = self.start_epoch
        else:
            self.start_epoch = 0

        self.criterion = LossFuncs(train_size=params.Training.train_size[0])

        self.valid_data = self.DataGen.gen_valid_data()

        self.recorder = {}
        self.init_recorder()

        self.no_improve = 0  # 7次无变化 终止
        self.best_loss = np.nan
        self.best_jaccard = np.nan

        self.params = params
        save_yaml(params, params.Training.result_path + 'train_params.yaml')
        create_infer_yaml(params, params.Training.result_path + 'infer_params.yaml')

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

    def train(self):

        print('start training!')

        while self.start_epoch < self.params.Training.max_epoch:
            if self.no_improve == 7:
                break

            tot_cost = []
            self.loss_best = np.inf
            tt = time.time()
            local_context = True
            for i in range(0, self.params.Training.eval_iteration):

                locs, X, Y, Z, I, s_mask, xyzi_gt = self.DataGen.generate_batch_newest(self.params.Training.batch_size, local_context=local_context)

                imgs_sim = self.DataGen.simulate_image(s_mask, xyzi_gt, locs, X, Y, Z, I)

                if local_context:
                    imgs_sim = imgs_sim.reshape([self.params.Training.batch_size, 3, locs.shape[-2], locs.shape[-1]])
                    mid_frame = torch.arange(1, locs.shape[0], 3)
                    xyzi_gt = xyzi_gt[mid_frame]
                    s_mask = s_mask[mid_frame]
                    locs = locs[mid_frame]

                p, xyzi_est, xyzi_sig = self.network.forward(imgs_sim, test=False)

                loss_total = self.criterion.final_loss(p, xyzi_est, xyzi_sig, xyzi_gt, s_mask, locs)

                self.optimizer.zero_grad()

                loss_total.backward()
                # avoid too large gradient
                torch.nn.utils.clip_grad_norm_(list(self.network.parameters()), max_norm=0.03, norm_type=2)

                # update the network and the optimizer state
                self.optimizer.step()
                self.scheduler.step()

                tot_cost.append(cpu(loss_total))

            self.start_epoch += 1
            print(f"Epoch{self.start_epoch}/{self.params.Training.max_epoch}")
            self.recorder['cost_hist'][self.start_epoch] = np.mean(tot_cost)
            self.recorder['update_time'][self.start_epoch] = (time.time() - tt) * 1000 / self.params.Training.eval_iteration

            self.evaluation()

            print('{}{}{:0.3f}'.format(' || ', 'Cost: ', self.recorder['cost_hist'][self.start_epoch]), end='')
            print('{}{}{}'.format(' || ', 'BatchNr.: ', self.params.Training.eval_iteration * (self.start_epoch)), end='')
            print('{}{}{:0.1f}{}'.format(' || ', 'Time Upd.: ', self.recorder['update_time'][self.start_epoch], ' ms '))

        print('training finished!')

    def evaluation(self):
        self.network.eval()
        loss = 0
        pred_list = []
        truth_list = []

        with torch.set_grad_enabled(False):
            for batch_ind, (xemit, yemit, z, S, Nphotons, s_mask, gt, img_sim) in enumerate(
                    self.valid_data):

                P, xyzi_est, xyzi_sig = self.network.forward(img_sim, test=True)
                gt, s_mask, S = gt[:, 1:-1], s_mask[:, 1:-1], S[:, 1:-1]
                loss_total = self.criterion.final_loss(P, xyzi_est, xyzi_sig, gt, s_mask, S)

                pred, match = self.EvalMetric.predlist(P, xyzi_est, gt, batch_ind)

                pred_list = pred_list + pred
                truth_list = truth_list + match

                loss = loss + loss_total

        if loss/self.params.Training.valid_frame_num < self.loss_best:
            self.loss_best = loss/self.params.Training.valid_frame_num
            self.save_model()

        pred_dict, match = self.EvalMetric.limited_matching(truth_list, pred_list)
        for k in self.recorder.keys():
            if k in pred_dict:
                self.recorder[k][self.start_epoch] = pred_dict[k]

    def save_model(self):
        if not (os.path.isdir(self.params.Training.result_path)):
            os.mkdir(self.params.Training.result_path)
        path_checkpoint = self.params.Training.result_path + 'checkpoint.pkl'
        torch.save(self, path_checkpoint)

    def analyze(self, im, test=True):
        p, xyzi_est, xyzi_sig = self.network.forward(im, test=test)
        infer_dict = self.network.post_process(p, xyzi_est)

        return infer_dict




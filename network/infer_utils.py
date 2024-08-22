import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import trange
from torch.cuda.amp import autocast as autocast
import numpy as np
from PSFLocModel import *

from utils.local_tifffile import *
from utils.record_utils import *

class InferDataset(Dataset):
    # initialization of the dataset
    def __init__(self, tif_file,win_size,padding):

        self.tif_file = TiffFile(tif_file, is_ome=True)
        self.total_shape = self.tif_file.series[0].shape
        self.data_info = self.get_img_info(self.total_shape,win_size)
        self.win_size =win_size
        self.padding = padding

        # total number of samples in the dataset
    def __len__(self):
        return len(self.data_info)

    # sampling one example from the data
    def __getitem__(self, index):
        # select sample

        frame_index,fov_coord = self.data_info[index]
        end_coord=[]
        img_end_coord = []
        start_coord = []
        fov_start = []

        if fov_coord[0]-self.padding <= 0:
            start_coord.append(self.padding)
            fov_start.append(fov_coord[0])
        else:
            start_coord.append(0)
            fov_start.append(fov_coord[0]-self.padding)

        if fov_coord[1]-self.padding <= 0:
            start_coord.append(self.padding)
            fov_start.append(fov_coord[1])
        else:
            start_coord.append(0)
            fov_start.append(fov_coord[1]-self.padding)

        if fov_coord[0]+self.win_size + self.padding >= self.total_shape[-2]:
            end_coord.append(self.total_shape[-2]-fov_coord[0])
            img_end_coord.append(self.total_shape[-2])
        else:
            end_coord.append(self.win_size+2*self.padding)
            img_end_coord.append(fov_coord[0]+self.win_size+self.padding)

        if fov_coord[1]+self.win_size +self.padding >= self.total_shape[-1]:
            end_coord.append(self.total_shape[-1]-fov_coord[1])
            img_end_coord.append(self.total_shape[-1])
        else:
            end_coord.append(self.win_size+2*self.padding)
            img_end_coord.append(fov_coord[1]+self.win_size+self.padding)


        img_target = np.array(self.tif_file.asarray(key=frame_index,series=0),dtype = np.float32)
        img = np.zeros((self.win_size+2*self.padding, self.win_size+2*self.padding),dtype = np.float32)
        img[start_coord[0]:end_coord[0],start_coord[1]:end_coord[1]] = img_target[fov_start[0]:img_end_coord[0],
                                                          fov_start[1]:img_end_coord[1]]
        return frame_index,np.array(fov_coord),img

    @staticmethod
    def get_img_info(total_shape,win_size):
        data_info = []
        for i in range(total_shape[0]):
            for j in range(int(np.ceil(total_shape[-1]/win_size))):
                for k in range(int(np.ceil(total_shape[-2]/win_size))):
                    data_info.append((i,[j*win_size,k*win_size]))
        return data_info


class PsfInfer:
    def __init__(self,infer_par,eval_params, net_params, offset=0, factor=0, fd=False):
        self.win_size = infer_par['win_size']
        self.batch_size = eval_params['batch_size']
        self.padding = infer_par['padding']
        self.result_name = infer_par['result_name']
        self.train_size = self.win_size + 2 * self.padding
        self.infer_loader = DataLoader(dataset=InferDataset(infer_par['img_path'], self.win_size, self.padding),
                                       batch_size=self.batch_size)
        if fd:
            self.model = FdDeeploc(net_params)
        else:
            self.model = LocalizationCNN_Unet_downsample_128_Unet(True)
        self.model.cuda()
        self.loadmodel(infer_par['net_path'])
        self.EVAla = Eval(eval_params)
        self.start_ind = check_csv(self.result_name)
        self.gt = infer_par['gt_path']

    def loadmodel(self,path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['state_dict'])
        print(checkpoint['epoch'])
        print("=> loaded checkpoint")

    def inferdata(self):
        time_forward = 0
        time_temp = time.time()
        self.model.eval()
        ll = len(self.infer_loader)
        with trange(ll) as t:
            with torch.set_grad_enabled(False):
                with autocast():
                    for ind, (index, coord, img) in enumerate(self.infer_loader, start=self.start_ind):
                        img = img.reshape([-1, 1, self.train_size, self.train_size])
                        img_gpu = img.cuda()
                        torch.cuda.synchronize()
                        time0 = time.time()
                        P, xyzi_est, xyzi_sig = self.model(img_gpu)
                        torch.cuda.synchronize()
                        time_forward = time_forward + (time.time() - time0) * 1000  # time for network forward pass (per frame, ms)
                        
                        time_postprocess = time.time()
                        pred_list, _ = self.EVAla.inferlist(P, xyzi_est, index, coord, self.padding, self.win_size)
                        # if np.any(np.isnan(np.array(pred_list))):
                        #     P, xyzi_est, xyzi_sig = self.model(img_gpu)
                        if len(pred_list):
                            write_csv(pred_list, self.result_name)
                        torch.cuda.empty_cache()
                        t.set_description(desc="Data %i" % ind)
                        t.set_postfix(pred_number=len(pred_list))
                        t.update(1)
                        torch.cuda.empty_cache()
                        time_postprocess = (time.time() - time_postprocess) * 1000 / self.batch_size  # time for post-process (per frame, ms)
        time_temp = (time.time() - time_temp) * 1000
        time_assess = time.time()
        perf_dict, matches = self.EVAla.assess(self.gt, self.result_name)
        time_assess = (time.time() - time_assess) * 1000
        print("predict finish, \n time for network forward pass is " + str(time_forward) + "ms, \n" + "time for post-process is "
              + str(time_postprocess) + "ms, \n" + "time for assessment(per frame) is " + str(time_assess)+ "ms" )
        print("time for forward and post-process is " + str(time_temp) + "ms.")
        return pred_list, matches, perf_dict

    def inferdata_faster(self):
        time_temp = time.time()
        self.model.eval()
        pred_list = []
        ll = len(self.infer_loader)
        res = {}
        self.EVAla.limited_x = [self.padding * self.EVAla.x_scale, (self.win_size + self.padding) * self.EVAla.x_scale]
        self.EVAla.limited_y = [self.padding * self.EVAla.x_scale, (self.win_size + self.padding) * self.EVAla.y_scale]
        with trange(ll) as t:
            with torch.set_grad_enabled(False):
                with autocast():
                    for ind, (index, coord, img) in enumerate(self.infer_loader, start=self.start_ind):
                        img = img.reshape([-1, 1, self.train_size, self.train_size])
                        img_gpu = img.cuda()
                        time_forward = time.time()
                        P, xyzi_est, _ = self.model(img_gpu)
                        time_forward = (
                                                   time.time() - time_forward) * 1000 / self.batch_size  # time for network forward pass (per frame, ms)

                        time_postprocess = time.time()
                        # res[ind] = {
                        #     "index": index,
                        #     "coord": coord - self.padding,
                        #     "Prob": P,
                        #     "preds": xyzi_est
                        # }
                        pred_list += self.EVAla.predlist_faster(P, xyzi_est, ind ,None,  [0, 0])

                        t.set_description(desc="Data %i" % ind)
                        t.update(1)
                        time_postprocess = (
                                                       time.time() - time_postprocess) * 1000 / self.batch_size  # time for post-process (per frame, ms)
        time_temp = (time.time() - time_temp) * 1000
        time_assess = time.time()
        if self.gt != "":
            # perf_dict, pred_lists = self.EVAla.assess_faster(self.gt, self.result_name)
            # test_list=[]
            # with open(self.gt, 'r') as csvfile:
            #     reader = csv.reader(csvfile, delimiter=',')
            #     next(reader)
            #     for row in reader:
            #         test_list.append([float(r) for r in row])
            perf_dict, matches = self.EVAla.assess_faster(self.gt, pred_list)
        else:
            pred_lists = self.EVAla.inferlist_faster(res)
        time_assess = (time.time() - time_assess) * 1000 / (ll * self.batch_size)
        print("predict finish, \n time for network forward pass(per frame) is " + str(
            time_forward) + "ms, \n" + "time for post-process(per frame) is "
              + str(time_postprocess) + "ms, \n" + "time for assessment(per frame) is " + str(time_assess) + "ms")
        print("time for forward and post-process is " + str(time_temp) + "ms.")

    def inferdata_NPC(self):
        self.model.eval()
        ll = len(self.infer_loader)
        t_network = 0
        t_postprocess = 0
        with trange(ll) as t:
            with torch.set_grad_enabled(False):
                with autocast():
                    for ind, (index, coord, img) in enumerate(self.infer_loader, start=self.start_ind):
                        img = img.reshape([-1, 1, self.train_size, self.train_size])
                        img_gpu = img.cuda()
                        t0 = time.time()
                        P, xyzi_est, xyzi_sig = self.model(img_gpu)
                        t_network = t_network + time.time() - t0

                        t1 = time.time()
                        pred_list, _ = self.EVAla.inferlist(P, xyzi_est, index, coord, self.padding,
                                                            self.win_size)
                        if len(pred_list):
                            write_csv(pred_list, self.result_name)
                        torch.cuda.empty_cache()
                        t.set_description(desc="Data %i" % ind)
                        t.set_postfix(pred_number=len(pred_list))
                        t.update(1)
                        torch.cuda.empty_cache()
                        t_postprocess = t_postprocess + time.time() - t1

        t3 = time.time()
        t_assess = time.time() - t3
        print("predict finish, \n time of network is " + str(t_network) + "time of postprocess is " + str(
            t_postprocess) + "time of assess is " + str(t_assess))


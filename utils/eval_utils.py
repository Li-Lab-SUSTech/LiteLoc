import torch
import torch.nn.functional as func

import csv
import numpy as np
import copy
import scipy.io as scio
from scipy.spatial.distance import cdist
from utils.help_utils import cpu, gpu, flip_filt

class EvalMetric:
    def __init__(self, psf_params, train_params):
        self.threshold = 0.3
        self.candi_thre = 0.3
        self.tolerance = 250
        self.tolerance_ax = 500
        self.batch_size = train_params.batch_size
        self.min_int = train_params.photon_range[0] / train_params.photon_range[1]
        self.int_scale = train_params.photon_range[1]
        self.z_scale = psf_params.z_scale
        if psf_params.simulate_method == 'vector':
            self.limited_x = [0, psf_params.vector_psf.pixelSizeX * train_params.train_size[0]]
            self.limited_y = [0, psf_params.vector_psf.pixelSizeY * train_params.train_size[1]]
            self.x_scale = psf_params.vector_psf.pixelSizeX
            self.y_scale = psf_params.vector_psf.pixelSizeY
        else:  # todo: load params from spline file
            calibration_info = scio.loadmat(psf_params.spline_psf.calibration_file, struct_as_record=False, squeeze_me=True)['SXY']  # todo: there is a bug
            self.limited_x = [0, calibration_info.zernikefit.pixelSizeX * train_params.train_size[0]]
            self.limited_y = [0, calibration_info.zernikefit.pixelSizeY * train_params.train_size[1]]
            self.x_scale = calibration_info.zernikefit.pixelSizeX
            self.y_scale = calibration_info.zernikefit.pixelSizeY

    def nms_func(self,p, candi_thre=0.3, xo=None, yo=None, zo=None):
        with torch.no_grad():
            diag = 0  # 1/np.sqrt(2)

            p_copy = p + 0

            # probability values > 0.3 are regarded as possible locations

            # p_clip = torch.where(p > 0.3, p, torch.zeros_like(p))[:, None]
            p_clip = torch.where(p > candi_thre, p, torch.zeros_like(p))[:, None]  # fushuang

            # localize maximum values within a 3x3 patch

            pool = func.max_pool2d(p_clip, 3, 1, padding=1)
            max_mask1 = torch.eq(p[:, None], pool).float()

            # Add probability values from the 4 adjacent pixels

            filt = np.array([[diag, 1, diag], [1, 1, 1], [diag, 1, diag]], ndmin=4)
            conv = func.conv2d(p[:, None], gpu(filt), padding=1)
            p_ps1 = max_mask1 * conv

            # In order do be able to identify two fluorophores in adjacent pixels we look for probablity values > 0.6 that are not part of the first mask

            p_copy *= (1 - max_mask1[:, 0])
            p_clip = torch.where(p_copy > 0.6, p_copy, torch.zeros_like(p_copy))[:, None]  # fushuang
            max_mask2 = torch.where(p_copy > 0.6, torch.ones_like(p_copy), torch.zeros_like(p_copy))[:,
                        None]  # fushuang
            p_ps2 = max_mask2 * conv

            # This is our final clustered probablity which we then threshold (normally > 0.7) to get our final discrete locations
            p_ps = p_ps1 + p_ps2

            if xo is None:
                return p_ps[:, 0].cpu()

            max_mask = torch.clamp(max_mask1 + max_mask2, 0, 1)

            mult_1 = max_mask1 / p_ps1
            mult_1[torch.isnan(mult_1)] = 0
            mult_2 = max_mask2 / p_ps2
            mult_2[torch.isnan(mult_2)] = 0

            # The rest is weighting the offset variables by the probabilities

            z_mid = zo * p
            z_conv1 = func.conv2d((z_mid * (1 - max_mask2[:, 0]))[:, None], gpu(filt), padding=1)
            z_conv2 = func.conv2d((z_mid * (1 - max_mask1[:, 0]))[:, None], gpu(filt), padding=1)

            zo_ps = z_conv1 * mult_1 + z_conv2 * mult_2
            zo_ps[torch.isnan(zo_ps)] = 0

            x_mid = xo * p
            x_mid_filt = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], ndmin=4)
            xm_conv1 = func.conv2d((x_mid * (1 - max_mask2[:, 0]))[:, None], gpu(x_mid_filt), padding=1)
            xm_conv2 = func.conv2d((x_mid * (1 - max_mask1[:, 0]))[:, None], gpu(x_mid_filt), padding=1)

            x_left = (xo + 1) * p
            x_left_filt = flip_filt(np.array([[diag, 0, 0], [1, 0, 0], [diag, 0, 0]], ndmin=4))
            xl_conv1 = func.conv2d((x_left * (1 - max_mask2[:, 0]))[:, None], gpu(x_left_filt), padding=1)
            xl_conv2 = func.conv2d((x_left * (1 - max_mask1[:, 0]))[:, None], gpu(x_left_filt), padding=1)

            x_right = (xo - 1) * p
            x_right_filt = flip_filt(np.array([[0, 0, diag], [0, 0, 1], [0, 0, diag]], ndmin=4))
            xr_conv1 = func.conv2d((x_right * (1 - max_mask2[:, 0]))[:, None], gpu(x_right_filt), padding=1)
            xr_conv2 = func.conv2d((x_right * (1 - max_mask1[:, 0]))[:, None], gpu(x_right_filt), padding=1)

            xo_ps = (xm_conv1 + xl_conv1 + xr_conv1) * mult_1 + (xm_conv2 + xl_conv2 + xr_conv2) * mult_2

            y_mid = yo * p
            y_mid_filt = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], ndmin=4)
            ym_conv1 = func.conv2d((y_mid * (1 - max_mask2[:, 0]))[:, None], gpu(y_mid_filt), padding=1)
            ym_conv2 = func.conv2d((y_mid * (1 - max_mask1[:, 0]))[:, None], gpu(y_mid_filt), padding=1)

            y_up = (yo + 1) * p
            y_up_filt = flip_filt(np.array([[diag, 1, diag], [0, 0, 0], [0, 0, 0]], ndmin=4))
            yu_conv1 = func.conv2d((y_up * (1 - max_mask2[:, 0]))[:, None], gpu(y_up_filt), padding=1)
            yu_conv2 = func.conv2d((y_up * (1 - max_mask1[:, 0]))[:, None], gpu(y_up_filt), padding=1)

            y_down = (yo - 1) * p
            y_down_filt = flip_filt(np.array([[0, 0, 0], [0, 0, 0], [diag, 1, diag]], ndmin=4))
            yd_conv1 = func.conv2d((y_down * (1 - max_mask2[:, 0]))[:, None], gpu(y_down_filt), padding=1)
            yd_conv2 = func.conv2d((y_down * (1 - max_mask1[:, 0]))[:, None], gpu(y_down_filt), padding=1)

            yo_ps = (ym_conv1 + yu_conv1 + yd_conv1) * mult_1 + (ym_conv2 + yu_conv2 + yd_conv2) * mult_2

            return cpu(p_ps[:, 0]), cpu(xo_ps[:, 0]), cpu(yo_ps[:, 0]), cpu(zo_ps[:, 0])

    def predlist(self, P, xyzi_est, target, start):
        start = start * self.batch_size
        xo = torch.squeeze(xyzi_est[:,0,:,:])
        yo = torch.squeeze(xyzi_est[:,1, :, :])
        zo = torch.squeeze(self.z_scale * xyzi_est[:,2,:,:])
        ints = np.squeeze(cpu(self.int_scale * xyzi_est[:,3,:,:]))
        p_nms = torch.squeeze(gpu(P))

        p_nms = self.nms_func(p_nms, candi_thre=self.candi_thre)
        xo = cpu(xo)
        yo = cpu(yo)
        zo = cpu(zo)
        ints = cpu(ints)

        sample = np.where(p_nms > self.threshold, 1, 0)

        # number of examples in the batch
        pred_list = []
        for i in range(len(xo)):
            pos = np.nonzero(sample[i])  # get the deterministic pixel position
            for j in range(len(pos[0])):
                pred_list.append([i+start,(0.5 + pos[1][j] + xo[i,pos[0][j], pos[1][j]]) * self.x_scale,
                                  (0.5 + pos[0][j] + yo[i,pos[0][j], pos[1][j]]) * self.y_scale,
                                  zo[i,pos[0][j], pos[1][j]],
                                  ints[i,pos[0][j], pos[1][j]],
                                  p_nms[i,pos[0][j], pos[1][j]]])
        test_list = []
        target = np.squeeze(cpu(target))
        for i in range(len(target)):
            pos = np.nonzero(target[i])  # get the deterministic pixel position
            for j in range(0,len(pos[0]),4):
                test_list.append([i+start, target[i,pos[0][j], pos[1][j]]* self.x_scale,
                                  target[i,pos[0][j+1], pos[1][j+1]] * self.y_scale,
                                  target[i,pos[0][j+2], pos[1][j+2]]*self.z_scale,
                                  target[i,pos[0][j+3], pos[1][j+3]]*self.int_scale])
        return pred_list,test_list

    def limited_matching(self,truth_origin, pred_list_origin):

        matches = []
        perf_dict = {'recall': 0, 'precision': 0, 'jaccard': 0,  'rmse_lat': 0,
                     'rmse_ax': 0,'rmse_vol': 0, 'jor': 0, 'eff_lat':0, 'eff_ax':0,
                     'eff_3d': 0}
        truth = copy.deepcopy(truth_origin)
        pred_list = copy.deepcopy(pred_list_origin)

        truth_array = np.array(truth)
        pred_array = np.array(pred_list)
        if len(pred_list) == 0:
            print('after FOV segmentation, pred_list is empty!')
            return perf_dict, matches

        # filter prediction and gt according to limited_x;y
        t_inds = np.where(
            (truth_array[:, 1] < self.limited_x[0]) | (truth_array[:, 1] > self.limited_x[1]) |
            (truth_array[:, 2] < self.limited_y[0]) | (truth_array[:, 2] > self.limited_y[1]))
        p_inds = np.where(
            (pred_array[:, 1] < self.limited_x[0]) | (pred_array[:, 1] > self.limited_x[1]) |
            (pred_array[:, 2] < self.limited_y[0]) | (pred_array[:, 2] > self.limited_y[1]))
        for t in reversed(t_inds[0]):
            del (truth[t])
        for p in reversed(p_inds[0]):
            del (pred_list[p])

        if len(pred_list) == 0:
            print('after border, pred_list is empty!')
            return perf_dict, matches

        print('{}{}{}{}{}'.format('after FOV and border segmentation,'
                                   , 'truth: ', len(truth), ' ,preds: ', len(pred_list)))

        TP = 0
        FP = 0.0001
        FN = 0.0001
        MSE_lat = 0
        MSE_ax = 0
        MSE_vol = 0

        for i in range(0, int(truth_origin[-1][0]) + 1):  # traverse all gt frames

            tests = []  # gt in each frame
            preds = []  # prediction in each frame

            if len(truth) > 0:  # after border filtering and area segmentation, truth could be empty
                while truth[0][0] == i:
                    tests.append(truth.pop(0))  # put all gt in the tests
                    if len(truth) < 1:
                        break
            if len(pred_list) > 0:
                while pred_list[0][0] == i:
                    preds.append(pred_list.pop(0))  # put all predictions in the preds
                    if len(pred_list) < 1:
                        break

            # if preds is empty, it means no detection on the frame, all tests are FN
            if len(preds) == 0:
                FN += len(tests)
                continue  # no need to calculate metric
            # if the gt of this frame is empty, all preds on this frame are FP
            if len(tests) == 0:
                FP += len(preds)
                continue  # no need to calculate metric

            # calculate the Euclidean distance between all gt and preds, get a matrix [number of gt, number of preds]
            dist_arr = cdist(np.array(tests)[:, 1:3], np.array(preds)[:, 1:3])
            ax_arr = cdist(np.array(tests)[:, 3:4], np.array(preds)[:, 3:4])
            tot_arr = np.sqrt(dist_arr ** 2 + ax_arr ** 2)

            if self.tolerance_ax == np.inf:
                tot_arr = dist_arr

            match_tests = copy.deepcopy(tests)
            match_preds = copy.deepcopy(preds)

            if dist_arr.size > 0:
                while np.min(dist_arr) < self.tolerance:
                    r, c = np.where(tot_arr == np.min(tot_arr))  # select the positions pair with shortest distance
                    r = r[0]
                    c = c[0]
                    if ax_arr[r, c] < self.tolerance_ax and dist_arr[
                        r, c] < self.tolerance:  # compare the distance and tolerance
                        if match_tests[r][-1] > self.min_int:  # photons should be larger than min_int

                            MSE_lat += dist_arr[r, c] ** 2
                            MSE_ax += ax_arr[r, c] ** 2
                            MSE_vol += dist_arr[r, c] ** 2 + ax_arr[r, c] ** 2
                            TP += 1
                            matches.append([match_tests[r][1], match_tests[r][2], match_tests[r][3], match_tests[r][4],
                                            match_preds[c][1], match_preds[c][2], match_preds[c][3], match_preds[c][4]])
                                            # match_preds[c][5]])

                        dist_arr[r, :] = np.inf
                        dist_arr[:, c] = np.inf
                        tot_arr[r, :] = np.inf
                        tot_arr[:, c] = np.inf

                        tests[r][-1] = -100  # photon cannot be negative, work as a flag
                        preds.pop()

                    dist_arr[r, c] = np.inf
                    tot_arr[r, c] = np.inf

            for j in reversed(range(len(tests))):
                if tests[j][-1] < self.min_int:  # delete matched gt
                    del (tests[j])

            FP += len(preds)  # all remaining preds are FP
            FN += len(tests)  # all remaining gt are FN



        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        jaccard = TP / (TP + FP + FN)
        rmse_lat = np.sqrt(MSE_lat / ((TP + 0.00001)*2))
        rmse_ax = np.sqrt(MSE_ax / (TP + 0.00001))
        rmse_vol = np.sqrt(MSE_vol / (TP + 0.00001))
        jor = 100 * jaccard / rmse_lat

        eff_lat = 100 - np.sqrt((100 - 100 * jaccard) ** 2 + 1 ** 2 * rmse_lat ** 2)
        eff_ax = 100 - np.sqrt((100 - 100 * jaccard) ** 2 + 0.5 ** 2 * rmse_ax ** 2)
        eff_3d = (eff_lat + eff_ax) / 2

        matches = np.array(matches)

        if len(matches)==0:
            print('matches is empty!')
            return perf_dict, matches



        perf_dict = {'recall': recall, 'precision': precision, 'jaccard': jaccard, 'rmse_lat': rmse_lat,
                     'rmse_ax': rmse_ax, 'rmse_vol': rmse_vol,'jor': jor, 'eff_lat': eff_lat, 'eff_ax': eff_ax,
                     'eff_3d': eff_3d}
        print('{}{:0.3f}'.format('Recall: ', recall))
        print('{}{:0.3f}'.format('Precision: ', precision))
        print('{}{:0.3f}'.format('Jaccard: ', 100 * jaccard))
        print('{}{:0.3f}'.format('RMSE_lat: ', rmse_lat))
        print('{}{:0.3f}'.format('RMSE_ax: ', rmse_ax))
        print('{}{:0.3f}'.format('RMSE_vol: ', rmse_vol))
        print('{}{:0.3f}'.format('Jaccard/RMSE: ', jor))
        print('{}{:0.3f}'.format('Eff_lat: ', eff_lat))
        print('{}{:0.3f}'.format('Eff_ax: ', eff_ax))
        print('{}{:0.3f}'.format('Eff_3d: ', eff_3d))
        print('FN: ' + str(np.round(FN)) + ' FP: ' + str(np.round(FP)))

        return perf_dict, matches

    
def limited_matching(truth_origin, pred_list_origin, eval_params):
    matches = []
    perf_dict = {'recall': 0, 'precision': 0, 'jaccard': 0, 'rmse_lat': 0,
                 'rmse_ax': 0, 'rmse_vol': 0, 'jor': 0, 'eff_lat': 0, 'eff_ax': 0,
                 'eff_3d': 0}
    truth = copy.deepcopy(truth_origin)
    pred_list = copy.deepcopy(pred_list_origin)

    truth_array = np.array(truth)
    pred_array = np.array(pred_list)
    if len(pred_list) == 0:
        print('after FOV segmentation, pred_list is empty!')
        return perf_dict, matches

    # filter prediction and gt according to limited_x;y
    t_inds = np.where(
        (truth_array[:, 1] < eval_params['limited_x'][0]) | (truth_array[:, 1] > eval_params['limited_x'][1]) |
        (truth_array[:, 2] < eval_params['limited_y'][0]) | (truth_array[:, 2] > eval_params['limited_y'][1]))
    p_inds = np.where(
        (pred_array[:, 1] < eval_params['limited_x'][0]) | (pred_array[:, 1] > eval_params['limited_x'][1]) |
        (pred_array[:, 2] < eval_params['limited_y'][0]) | (pred_array[:, 2] > eval_params['limited_y'][1]))
    for t in reversed(t_inds[0]):
        del (truth[t])
    for p in reversed(p_inds[0]):
        del (pred_list[p])

    if len(pred_list) == 0:
        print('after border, pred_list is empty!')
        return perf_dict, matches

    print('{}{}{}{}{}'.format('after FOV and border segmentation,'
                              , 'truth: ', len(truth), ' ,preds: ', len(pred_list)))

    TP = 0
    FP = 0.0001
    FN = 0.0001
    MSE_lat = 0
    MSE_ax = 0
    MSE_vol = 0

    for i in range(0, int(truth_origin[-1][0]) + 1):  # traverse all gt frames

        tests = []  # gt in each frame
        preds = []  # prediction in each frame

        if len(truth) > 0:  # after border filtering and area segmentation, truth could be empty
            while truth[0][0] == i:
                tests.append(truth.pop(0))  # put all gt in the tests
                if len(truth) < 1:
                    break
        if len(pred_list) > 0:
            while pred_list[0][0] == i:
                preds.append(pred_list.pop(0))  # put all predictions in the preds
                if len(pred_list) < 1:
                    break

        # if preds is empty, it means no detection on the frame, all tests are FN
        if len(preds) == 0:
            FN += len(tests)
            continue  # no need to calculate metric
        # if the gt of this frame is empty, all preds on this frame are FP
        if len(tests) == 0:
            FP += len(preds)
            continue  # no need to calculate metric

        # calculate the Euclidean distance between all gt and preds, get a matrix [number of gt, number of preds]
        dist_arr = cdist(np.array(tests)[:, 1:3], np.array(preds)[:, 1:3])
        ax_arr = cdist(np.array(tests)[:, 3:4], np.array(preds)[:, 3:4])
        tot_arr = np.sqrt(dist_arr ** 2 + ax_arr ** 2)

        if eval_params['tolerance_ax'] == np.inf:
            tot_arr = dist_arr

        match_tests = copy.deepcopy(tests)
        match_preds = copy.deepcopy(preds)

        if dist_arr.size > 0:
            while np.min(dist_arr) < eval_params['tolerance']:
                r, c = np.where(tot_arr == np.min(tot_arr))  # select the positions pair with shortest distance
                r = r[0]
                c = c[0]
                if ax_arr[r, c] < eval_params['tolerance_ax'] and dist_arr[
                    r, c] < eval_params['tolerance']:  # compare the distance and tolerance
                    if match_tests[r][-1] > eval_params['min_int']:  # photons should be larger than min_int

                        MSE_lat += dist_arr[r, c] ** 2
                        MSE_ax += ax_arr[r, c] ** 2
                        MSE_vol += dist_arr[r, c] ** 2 + ax_arr[r, c] ** 2
                        TP += 1
                        matches.append([match_tests[r][1], match_tests[r][2], match_tests[r][3], match_tests[r][4],
                                        match_preds[c][1], match_preds[c][2], match_preds[c][3], match_preds[c][4]])
                        # match_preds[c][5]])

                    dist_arr[r, :] = np.inf
                    dist_arr[:, c] = np.inf
                    tot_arr[r, :] = np.inf
                    tot_arr[:, c] = np.inf

                    tests[r][-1] = -100  # photon cannot be negative, work as a flag
                    preds.pop()

                dist_arr[r, c] = np.inf
                tot_arr[r, c] = np.inf

        for j in reversed(range(len(tests))):
            if tests[j][-1] < eval_params['min_int']:  # delete matched gt
                del (tests[j])

        FP += len(preds)  # all remaining preds are FP
        FN += len(tests)  # all remaining gt are FN

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    jaccard = TP / (TP + FP + FN)
    rmse_lat = np.sqrt(MSE_lat / ((TP + 0.00001) * 2))
    rmse_ax = np.sqrt(MSE_ax / (TP + 0.00001))
    rmse_vol = np.sqrt(MSE_vol / (TP + 0.00001))
    jor = 100 * jaccard / rmse_lat

    eff_lat = 100 - np.sqrt((100 - 100 * jaccard) ** 2 + 1 ** 2 * rmse_lat ** 2)
    eff_ax = 100 - np.sqrt((100 - 100 * jaccard) ** 2 + 0.5 ** 2 * rmse_ax ** 2)
    eff_3d = (eff_lat + eff_ax) / 2

    matches = np.array(matches)

    if len(matches) == 0:
        print('matches is empty!')
        return perf_dict, matches

    perf_dict = {'recall': recall, 'precision': precision, 'jaccard': jaccard, 'rmse_lat': rmse_lat,
                 'rmse_ax': rmse_ax, 'rmse_vol': rmse_vol, 'jor': jor, 'eff_lat': eff_lat, 'eff_ax': eff_ax,
                 'eff_3d': eff_3d}
    print('{}{:0.3f}'.format('Recall: ', recall))
    print('{}{:0.3f}'.format('Precision: ', precision))
    print('{}{:0.3f}'.format('Jaccard: ', 100 * jaccard))
    print('{}{:0.3f}'.format('RMSE_lat: ', rmse_lat))
    print('{}{:0.3f}'.format('RMSE_ax: ', rmse_ax))
    print('{}{:0.3f}'.format('RMSE_vol: ', rmse_vol))
    print('{}{:0.3f}'.format('Jaccard/RMSE: ', jor))
    print('{}{:0.3f}'.format('Eff_lat: ', eff_lat))
    print('{}{:0.3f}'.format('Eff_ax: ', eff_ax))
    print('{}{:0.3f}'.format('Eff_3d: ', eff_3d))
    print('FN: ' + str(np.round(FN)) + ' FP: ' + str(np.round(FP)))

    return perf_dict, matches

def assess_file(test_csv, pred_inp, eval_params):

    test_list = []
    count = 0
    if isinstance(test_csv, str):
        with open(test_csv, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            next(reader)
            for row in reader:
                test_list.append([float(r) for r in row])
                count = count + 1
    else:
        for r in test_csv:
            test_list.append([i for i in r])

    test_frame_nbr = test_list[-1][0]

    print('{}{}{}{}{}'.format('\nevaluation on ', test_frame_nbr,
                              ' images, ', 'contain ground truth: ', len(test_list)), end='')

    pred_list = []
    if isinstance(pred_inp, str):

        with open(pred_inp, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            next(reader)
            for row in reader:
                pred_list.append([float(r) for r in row])
    else:
        for r in pred_inp:
            pred_list.append([i for i in r])

    print('{}{}'.format(', preds:', len(pred_list)))

    perf_dict, matches = limited_matching(test_list, pred_list, eval_params)

    return perf_dict, matches

def assess_data(gt, pred, eval_params):

    test_list = []
    count = 0
    if isinstance(gt, str):
        with open(gt, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            next(reader)
            for row in reader:
                test_list.append([float(r) for r in row])
                count = count + 1
    else:
        for r in gt:
            test_list.append([i for i in r])

    test_frame_nbr = test_list[-1][0]

    print('{}{}{}{}{}'.format('\nevaluation on ', test_frame_nbr,
                              ' images, ', 'contain ground truth: ', len(test_list)), end='')

    pred_list = []
    if isinstance(pred, str):

        with open(pred, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            next(reader)
            for row in reader:
                pred_list.append([float(r) for r in row])
    else:
        for r in pred:
            pred_list.append([i for i in r])

    print('{}{}'.format(', preds:', len(pred_list)))

    perf_dict, matches = limited_matching(test_list, pred_list, eval_params)

    return perf_dict, matches
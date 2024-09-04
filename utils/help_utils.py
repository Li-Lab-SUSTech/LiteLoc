import numpy as np
import torch
import torch.nn as nn
import scipy.stats
import yaml
import recursivenamespace
from matplotlib import pyplot as plt
import random
import sys
import datetime
import os
import pickle
import csv
from tifffile import TiffFile

def load_yaml(yaml_file):
    with open(yaml_file, 'r') as f:
        params = yaml.load(f, Loader=yaml.SafeLoader)
    return recursivenamespace.RecursiveNamespace(**params)

def writelog(log_path):
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    # 日志文件名按照程序运行时间设置
    log_file_name = log_path + 'log-' + datetime.datetime.now().strftime('%Y-%m-%d') + '.log'
    # 记录正常的 print 信息
    sys.stdout = Logger(log_file_name)
    # 记录 traceback 异常信息
    sys.stderr = Logger(log_file_name)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True  # this seems affect the speed of some networks
    torch.backends.cudnn.benchmark = False

def gpu(x, data_type=torch.float32):
    """
    Transforms numpy array or torch tensor to torch.cuda.FloatTensor
    """

    if not isinstance(x, torch.Tensor):
        return torch.tensor(x, device='cuda:0', dtype=data_type)
    return x.to(device='cuda:0', dtype=data_type)

def cpu(x, data_type=np.float32):
    """
    Transforms torch tensor into numpy array
    """

    if not isinstance(x, torch.Tensor):
        return np.array(x, dtype=data_type)
    return x.cpu().detach().numpy().astype(data_type)

def flip_filt(filt):
    '''Returns filter flipped over x and y dimension'''
    return np.ascontiguousarray(filt[...,::-1,::-1])

def get_bg_stats(images, percentile=10, plot=False, xlim=None, floc=0):
    """Infers the parameters of a gamma distribution that fit the background of SMLM recordings.
    Identifies the darkest pixels from the averaged images as background and fits a gamma distribution to the histogram of intensity values.

    Parameters
    ----------
    images: array
        3D array of recordings
    percentile: float
        Percentile between 0 and 100. Sets the percentage of pixels that are assumed to only containg background activity (i.e. no fluorescent signal)
    plot: bool
        If true produces a plot of the histogram and fit
    xlim: list of floats
        Sets xlim of the plot
    floc: float
        Baseline for the the gamma fit. Equal to fitting gamma to (x - floc)

    Returns
    -------
    mean, scale: float
        Mean and scale parameter of the gamma fit
    """
    # 确保图片为正值
    ind = np.where(images <= 0)
    images[ind] = 1

    # 先将图像中每个位置求全部帧的平均，然后选出10%位置的值，然后得到小于这个值的位置的坐标
    map_empty = np.where(images.mean(0) < np.percentile(images.mean(0), percentile))
    # 取出imagestack中这些位置的所有值
    pixel_vals = images[:, map_empty[0], map_empty[1]].reshape(-1)
    # 调用scipy库的gamma拟合,返回的是alpha和scale=1/beta
    fit_alpha, fit_loc, fit_beta = scipy.stats.gamma.fit(pixel_vals, floc=floc)

    if plot:
        plt.figure(constrained_layout=True)
        if xlim is None:
            low, high = pixel_vals.min(), pixel_vals.max()
        else:
            low, high = xlim[0], xlim[1]

        _ = plt.hist(pixel_vals, bins=np.linspace(low, high), histtype='step', label='data')
        _ = plt.hist(np.random.gamma(shape=fit_alpha, scale=fit_beta, size=len(pixel_vals)) + floc,
                     bins=np.linspace(low, high), histtype='step', label='fit')
        plt.xlim(low, high)
        plt.legend()
        # plt.tight_layout()
        plt.show()
    return fit_alpha * fit_beta, fit_beta  # 返回gamma分布的期望

def read_first_size_gb_tiff(image_path, size_gb=4):
    with TiffFile(image_path, is_ome=False) as tif:  # todo: package load
        total_shape = tif.series[0].shape
        occu_mem = total_shape[0] * total_shape[1] * total_shape[2] * 16 / (1024 ** 3) / 8
        if occu_mem<size_gb:
            index_img=total_shape[0]
        else:
            index_img = int(size_gb/occu_mem*total_shape[0])
        images = tif.asarray(key=range(0,index_img), series=0)
    print("read first %d images" % (images.shape[0]))
    return images

def calculate_bg(image_path):
    images = read_first_size_gb_tiff(image_path)
    bg, _= get_bg_stats(images, percentile=50)
    return bg

class Logger(object):
    def __init__(self, file_name="Default.log", stream=sys.stdout):
        self.terminal = stream
        self.log = open(file_name, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def get_mean_percentile(images, percentile=10):
    """
    Returns the mean of the pixels at where their mean values are less than the given percentile of the average image

    Args:
        images (np.ndarray): 3D array of recordings
        percentile (float): Percentile between 0 and 100. Used to calculate the mean of the percentile of the images
    """

    idx_2d = np.where(images.mean(0) < np.percentile(images.mean(0), percentile))
    pixel_vals = images[:, idx_2d[0], idx_2d[1]]

    return pixel_vals.mean()

def place_psfs(psf_pars, W, S, ph_scale):

    recs = torch.zeros_like(S)
    h, w = S.shape[1], S.shape[2]
    # s_inds: [0, 0], [y2, y1], [x2, x1]
    s_inds = tuple(S.nonzero().transpose(1, 0))
    relu = nn.ReLU()
    # r_inds: [y2, x2], [y1, x1]
    r_inds = S.nonzero()[:, 1:]  # xy坐标
    uni_inds = S.sum(0).nonzero()

    x_rl = relu(uni_inds[:, 0] - psf_pars.psfSizeX // 2)  # y的位置
    y_rl = relu(uni_inds[:, 1] - psf_pars.psfSizeX // 2)  # x的位置

    x_wl = relu(psf_pars.psfSizeX // 2 - uni_inds[:, 0])
    x_wh = psf_pars.psfSizeX - (uni_inds[:, 0] + psf_pars.psfSizeX // 2 - h) - 1

    y_wl = relu(psf_pars.psfSizeX // 2 - uni_inds[:, 1])
    y_wh = psf_pars.psfSizeX - (uni_inds[:, 1] + psf_pars.psfSizeX // 2 - w) - 1

    r_inds_r = h * r_inds[:, 0] + r_inds[:, 1]
    uni_inds_r = h * uni_inds[:, 0] + uni_inds[:, 1]

    for i in range(len(uni_inds)):
        curr_inds = torch.nonzero(r_inds_r == uni_inds_r[i])[:, 0]
        w_cut = W[curr_inds, x_wl[i]: x_wh[i], y_wl[i]: y_wh[i]]

        recs[s_inds[0][curr_inds], x_rl[i]:x_rl[i] + w_cut.shape[1], y_rl[i]:y_rl[i] + w_cut.shape[2]] += w_cut

    return recs * ph_scale

def load_model(model_file):
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    return model

def ShowRecovery3D( match):
    # define a figure for 3D scatter plot
    ax = plt.axes(projection='3d')

    # plot boolean recoveries in 3D
    ax.scatter(match[:, 0], match[:, 1], match[:, 2], c='b', marker='o', label='GT', depthshade=False)
    ax.scatter(match[:, 4], match[:, 5], match[:, 6], c='r', marker='^', label='Rec', depthshade=False)

    # add labels and legend
    ax.set_xlabel('X [nm]')
    ax.set_ylabel('Y [nm]')
    ax.set_zlabel('Z [nm]')
    plt.legend()

def plot_points(x, y, image_size):
    plt.figure(dpi=400)
    plt.scatter(x, y, s=1, c='blue', alpha=0.7)
    plt.xticks([0, image_size])
    plt.yticks([0, image_size])
    # plt.axis('equal')
    plt.show()

def generate_pos(image_size, pixel_size, num_points, z_scale, save_path):
    # 设置图像大小
    center_size = int(image_size * 0.8)  # 中心正方形的大小占整个图像的95%, every side 0.05*image_size/2

    # 生成均匀分布的随机点
    x_center = np.random.uniform(-0.5 * center_size, 0.5 * center_size, num_points)
    y_center = np.random.uniform(-0.5 * center_size, 0.5 * center_size, num_points)
    cz_center = np.random.uniform(-z_scale, z_scale, num_points)

    # 移动坐标到图像中心
    x_center = (x_center + 0.5 * image_size) * pixel_size[0]
    y_center = (y_center + 0.5 * image_size) * pixel_size[1]

    # 将坐标保存到CSV文件
    with open(save_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # csvwriter.writerow(['X', 'Y', 'Z'])
        for row in zip(x_center, y_center, cz_center):
            csvwriter.writerow(row)
    plot_points(x_center / pixel_size[0], y_center / pixel_size[1], image_size)

def write_csv_array(input_array, filename, write_mode='write localizations'):
    """
    Writes a csv_file with different column orders depending on the input.
        [frame, x, y, z, photon, integrated prob, x uncertainty, y uncertainty,
         z uncertainty, photon uncertainty, x_offset, y_offset]

    Args:
        input_array (np.ndarray): molecule array that need to be written
        filename (str): path to csv_file
        write_mode (str):
            1. 'write paired localizations': write paired ground truth and predictions from the
                ailoc.common.assess.pair_localizations function, the format is
                ['frame', 'x_gt', 'y_gt', 'z_gt', 'photon_gt', 'x_pred', 'y_pred', 'z_pred', 'photon_pred'];

            2. 'write localizations': write predicted molecule list, the format is
                ['frame', 'xnm', 'ynm', 'znm', 'photon', 'prob', 'x_sig', 'y_sig', 'z_sig', 'photon_sig',
                'xo', 'yo'];

            3. 'append localizations': append to existing file using the format in 2;

            4. 'write rescaled localizations': write predicted molecule list with rescaled coordinates, the format is
                ['frame', 'xnm', 'ynm', 'znm', 'photon', 'prob', 'x_sig', 'y_sig', 'z_sig', 'photon_sig',
                'xo', 'yo', 'xo_rescale', 'yo_rescale', 'xnm_rescale', 'ynm_rescale'];
    """

    if write_mode == 'write paired localizations':
        with open(filename, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow(['frame', 'x_gt', 'y_gt', 'z_gt', 'photon_gt', 'x_pred', 'y_pred', 'z_pred',
                                'photon_pred'])
            for row in input_array:
                csvwriter.writerow([repr(element) for element in row])
    elif write_mode == 'write localizations':
        with open(filename, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow(['frame', 'xnm', 'ynm', 'znm', 'photon', 'prob', 'x_sig', 'y_sig', 'z_sig',
                                'photon_sig', 'xoffset', 'yoffset'])
            for row in input_array:
                csvwriter.writerow([repr(element) for element in row])
    elif write_mode == 'append localizations':
        with open(filename, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            for row in input_array:
                csvwriter.writerow([repr(element) for element in row])
    elif write_mode == 'write rescaled localizations':
        with open(filename, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow(['frame', 'xnm', 'ynm', 'znm', 'photon', 'prob', 'x_sig', 'y_sig', 'z_sig',
                                'photon_sig', 'xo', 'yo', 'xo_rescale', 'yo_rescale', 'xnm_rescale',
                                'ynm_rescale'])
            for row in input_array:
                csvwriter.writerow([repr(element) for element in row])
    else:
        raise ValueError('write_mode must be "write paired localizations", "write localizations", '
                         '"append localizations", or "write rescaled localizations"')
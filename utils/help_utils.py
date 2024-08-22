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
from torch.utils.data import Dataset
from utils.local_tifffile import *
import os
import pickle


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

def softp(x):
    '''Returns softplus(x)'''
    return(np.log(1+np.exp(x)))

def sigmoid(x):
    '''Returns sigmoid(x)'''
    return 1 / (1 + np.exp(-x))

def inv_softp(x):
    '''Returns inverse softplus(x)'''
    return np.log(np.exp(x)-1)

def inv_sigmoid(x):
    '''Returns inverse sigmoid(x)'''
    return -np.log(1/x-1)

def torch_arctanh(x):
    '''Returns arctanh(x) for tensor input'''
    return 0.5*torch.log(1+x) - 0.5*torch.log(1-x)

def torch_softp(x):
    '''Returns softplus(x) for tensor input'''
    return (torch.log(1+torch.exp(x)))

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
    with TiffFile(image_path, is_ome=False) as tif:
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


def writelog(log_path):
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    # 日志文件名按照程序运行时间设置
    log_file_name = log_path + 'log-' + datetime.datetime.now().strftime('%Y-%m-%d') + '.log'
    # 记录正常的 print 信息
    sys.stdout = Logger(log_file_name)
    # 记录 traceback 异常信息
    sys.stderr = Logger(log_file_name)

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

    x_rl = relu(uni_inds[:, 0] - psf_pars['Npixels'] // 2)  # y的位置
    y_rl = relu(uni_inds[:, 1] - psf_pars['Npixels'] // 2)  # x的位置

    x_wl = relu(psf_pars['Npixels'] // 2 - uni_inds[:, 0])
    x_wh = psf_pars['Npixels'] - (uni_inds[:, 0] + psf_pars['Npixels'] // 2 - h) - 1

    y_wl = relu(psf_pars['Npixels'] // 2 - uni_inds[:, 1])
    y_wh = psf_pars['Npixels'] - (uni_inds[:, 1] + psf_pars['Npixels'] // 2 - w) - 1

    r_inds_r = h * r_inds[:, 0] + r_inds[:, 1]
    uni_inds_r = h * uni_inds[:, 0] + uni_inds[:, 1]

    for i in range(len(uni_inds)):
        curr_inds = torch.nonzero(r_inds_r == uni_inds_r[i])[:, 0]
        w_cut = W[curr_inds, x_wl[i]: x_wh[i], y_wl[i]: y_wh[i]]

        recs[s_inds[0][curr_inds], x_rl[i]:x_rl[i] + w_cut.shape[1], y_rl[i]:y_rl[i] + w_cut.shape[2]] += w_cut

    return recs * ph_scale

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

def load_model(model_file):
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    return model

def load_yaml(yaml_file):
    with open(yaml_file, 'r') as f:
        params = yaml.load(f, Loader=yaml.SafeLoader)
    return recursivenamespace.RecursiveNamespace(**params)


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

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
from tqdm import tqdm
from scipy.fftpack import fft, fftshift
from torch.cuda.amp import autocast
import torch.nn.functional as F
from types import SimpleNamespace

def load_yaml(yaml_file):
    with open(yaml_file, 'r') as f:
        params = yaml.load(f, Loader=yaml.SafeLoader)
    return recursivenamespace.RecursiveNamespace(**params)

def dict_to_namespace(data):
    if isinstance(data, dict):
        return SimpleNamespace(**{key: dict_to_namespace(value) for key, value in data.items()})
    elif isinstance(data, list):
        return [dict_to_namespace(item) for item in data]
    else:
        return data

def namespace_to_dict(namespace):
    """
    将 SimpleNamespace 或者字典类型递归地转换为普通字典。
    """
    if isinstance(namespace, SimpleNamespace):
        return {key: namespace_to_dict(value) for key, value in namespace.__dict__.items()}
    elif isinstance(namespace, dict):
        return {key: namespace_to_dict(value) for key, value in namespace.items()}
    else:
        return namespace

def load_yaml_fy(yaml_file):
    with open(yaml_file, 'r') as f:
        params = yaml.load(f, Loader=yaml.SafeLoader)
    params = dict_to_namespace(params)
    return params

def save_yaml(params, yaml_file_path):
    params = namespace_to_dict(params)
    with open(yaml_file_path, 'w') as yaml_file:
        yaml.dump(params, yaml_file, sort_keys=False, default_flow_style=False)

def create_infer_yaml(params, yaml_file_path):
    if params.Training.infer_data is not None:
        infer_data = os.path.dirname(params.Training.infer_data) + '/'
    else:
        infer_data = os.path.join(os.path.expanduser('~'), params.Training.project_name) + 'results/'
    params = {
        'Loc_Model':{
            'model_path': str(params.Training.result_path) + 'checkpoint.pkl'
        },
        'Multi_Process':{
            'image_path': infer_data,
            'save_path': infer_data + 'result.csv',
            'time_block_gb': 1,
            'batch_size': 64,
            'sub_fov_size': 256,
            'over_cut': 8,
            'multi_gpu': True,
            'num_producers': 1
        }
    }
    with open(yaml_file_path, 'w') as yaml_file:
        yaml.dump(params, yaml_file, sort_keys=False, default_flow_style=False)

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

def calculate_bg(image_path, per=50):
    images = read_first_size_gb_tiff(image_path)
    bg, _= get_bg_stats(images, percentile=per)
    return bg

def calculate_factor_offset(image_path):
    images = read_first_size_gb_tiff(image_path)
    factor = images.mean(0).max().astype('float32')
    offset = images.mean().astype('float32')
    return factor, offset

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

def place_psfs(psf_pars, W, S, ph_scale): # todo: any two emitters must be in different pixels

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

def place_psfscc(psf_pars, W, S, ph_scale):

    recs = torch.zeros_like(S)
    h, w = S.shape[1], S.shape[2]

    s_inds = tuple(S.nonzero().transpose(1, 0))
    relu = nn.ReLU()

    r_inds = S.nonzero()[:, 1:]
    ans = r_inds
    for r in r_inds:
        tmp = S[0, r[0], r[1]]
        while tmp > 1:
            ans = torch.cat((ans, r.unsqueeze(0)), 0)
            s_inds = list(s_inds)
            s_inds[0] = torch.cat((s_inds[0], s_inds[0][0].unsqueeze(0)), 0)
            s_inds[1] = torch.cat((s_inds[1], r[0].unsqueeze(0)), 0)
            s_inds[2] = torch.cat((s_inds[2], r[1].unsqueeze(0)), 0)
            s_inds = tuple(s_inds)
            tmp -= 1

    r_inds = ans
    uni_inds = S.sum(0).nonzero()

    x_rl = relu(uni_inds[:, 0] - psf_pars.Npixels // 2)
    y_rl = relu(uni_inds[:, 1] - psf_pars.Npixels // 2)

    x_wl = relu(psf_pars.Npixels // 2 - uni_inds[:, 0])
    x_wh = psf_pars.Npixels - (uni_inds[:, 0] + psf_pars.Npixels // 2 - h) - 1
    x_wh = torch.where(x_wh > psf_pars.Npixels, psf_pars.Npixels, x_wh)

    y_wl = relu(psf_pars.Npixels // 2 - uni_inds[:, 1])
    y_wh = psf_pars.Npixels - (uni_inds[:, 1] + psf_pars.Npixels // 2 - w) - 1
    y_wh = torch.where(y_wh > psf_pars.Npixels, psf_pars.Npixels, y_wh)

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
    elif write_mode == 'write simple localizations':
        with open(filename, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow(['frame', 'xnm', 'ynm', 'znm', 'photon', 'prob'])
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

def calculate_fft_grid(molecule_list, image_size, pixel_size, fig_save_path=None): # molecule_list should be numpy.array type

    molecule_list = molecule_list[:, 1:5]

    super_res_factor = 10

    super_res_size = (image_size[0] * super_res_factor, image_size[1] * super_res_factor)
    super_res_image = np.zeros(super_res_size)

    # xnm_min = molecule_list[:, 0].min()
    # ynm_min = molecule_list[:, 1].max()

    for molecule in molecule_list:
        x_nm, y_nm, _, intensity = molecule
        x_super_res = int((x_nm) / (pixel_size / super_res_factor))
        y_super_res = int((y_nm) / (pixel_size / super_res_factor))

        if 0 <= x_super_res < super_res_size[0] and 0 <= y_super_res < super_res_size[1]:
            super_res_image[y_super_res, x_super_res] += 1 # intensity

    # plt.imshow(super_res_image)
    # plt.show()
    plt.imshow(super_res_image, cmap='hot', vmin=np.percentile(super_res_image, 2),
               vmax=np.percentile(super_res_image, 98))
    plt.show()

    target_freqs = 1 / pixel_size
    compressed_signal = np.sum(super_res_image, axis=0)
    fft_result = fft(compressed_signal)
    amplitude_spectrum = np.abs(fft_result)
    normalized_amplitude_spectrum = amplitude_spectrum / amplitude_spectrum[0]
    shifted_normalized_amplitude = fftshift(normalized_amplitude_spectrum)
    freqs = np.fft.fftfreq(len(compressed_signal), d=pixel_size / super_res_factor)
    shifted_freqs = fftshift(freqs)
    closest_index = np.argmin(np.abs(freqs - target_freqs))
    closest_freqs = freqs[closest_index]
    amplitude_at_closest_freqs = np.max(normalized_amplitude_spectrum[closest_index - 1:closest_index + 1])

    fig = plt.figure(dpi=300)
    plt.plot(shifted_freqs, shifted_normalized_amplitude)
    plt.scatter(closest_freqs, amplitude_at_closest_freqs, color='red', label='Most Prominent Peak')
    plt.text(closest_freqs, amplitude_at_closest_freqs,
             f'Freq: {closest_freqs:.4f}\nAmp: {amplitude_at_closest_freqs:.4f}', color='red', fontsize=10,
             ha='center', va='bottom', bbox=dict(facecolor='white', alpha=0.5))
    plt.title('Normalized Amplitude Spectrum (1D FFT)')
    plt.xlabel('Frequency ' + '$(nm^{-1})$')
    plt.ylabel('Normalized Amplitude')
    plt.legend()
    plt.grid()
    plt.show()
    if fig_save_path is not None:
        fig.savefig(fig_save_path, dpi=300)
    return closest_freqs, amplitude_at_closest_freqs


import copy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftshift
from scipy.signal import find_peaks


def compute_pixel_grid_idx_fy(molecule_list, pixel_size):

    image_size = [32, 32]
    super_res_factor = 10

    super_res_size = (image_size[0] * super_res_factor, image_size[1] * super_res_factor)
    super_res_image = np.zeros(super_res_size)

    for molecule in molecule_list:
        x_nm, y_nm, _, intensity = molecule
        x_super_res = int(x_nm / (pixel_size / super_res_factor))
        y_super_res = int(y_nm / (pixel_size / super_res_factor))

        if 0 <= x_super_res < super_res_size[0] and 0 <= y_super_res < super_res_size[1]:
            super_res_image[y_super_res, x_super_res] += intensity

    plt.imshow(super_res_image)
    plt.show()

    # fs = 1 / (pixel_size/super_res_factor)
    compressed_signal = np.sum(super_res_image, axis=0)
    fft_result = fft(compressed_signal)
    amplitude_spectrum = np.abs(fft_result)
    normalized_amplitude_spectrum = amplitude_spectrum / amplitude_spectrum[0]
    shifted_normalized_amplitude = fftshift(normalized_amplitude_spectrum)
    freqs = np.fft.fftfreq(len(compressed_signal))
    shifted_freqs = fftshift(freqs)

    peaks, properties = find_peaks(amplitude_spectrum[1:], height=0.1, prominence=0.05)

    if peaks.size > 0:
        most_prominent_peak_index = np.argmax(properties['prominences'])
        most_prominent_peak = peaks[most_prominent_peak_index]
        peak_amplitude = normalized_amplitude_spectrum[most_prominent_peak+1]
        peak_frequency = freqs[most_prominent_peak+1]

    plt.plot(shifted_freqs, shifted_normalized_amplitude)
    if peak_amplitude is not None:
        plt.scatter(freqs[most_prominent_peak + 1], normalized_amplitude_spectrum[most_prominent_peak + 1], color='red', label='Most Prominent Peak')
        plt.text(freqs[most_prominent_peak + 1], normalized_amplitude_spectrum[most_prominent_peak + 1],
                 f'Freq: {peak_frequency:.2f} Hz\nAmp: {peak_amplitude:.2f}', color='red', fontsize=10,
                 ha='center', va='bottom', bbox=dict(facecolor='white', alpha=0.5))
    plt.title('Normalized Amplitude Spectrum (1D FFT)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Normalized Amplitude')
    plt.legend()
    plt.grid()
    plt.show()


def get_hist_image(molecule_list, bin_size, image_size):
    image = np.zeros(image_size)
    for molecule in molecule_list:
        x_nm, y_nm, _, intensity = molecule
        x_idx = int(x_nm / bin_size)
        y_idx = int(y_nm / bin_size)

        if 0 <= x_idx < image_size[0] and 0 <= y_idx < image_size[1]:
            image[y_idx, x_idx] += intensity

    return image


def radial_sum(img):
    s = img.shape
    center = ((s[0]+1)//2, (s[1]+1)//2)
    rs = np.zeros(int(np.ceil(s[0]/2)+1))

    for k in range(s[0]):
        for l in range(s[1]):
            d = np.sqrt((k-center[0])**2 + (l-center[1])**2)
            ind = int(np.round(d))
            if ind < len(rs):
                rs[ind] += img[k, l]
    return rs


def make_2_blocks(block_size, molecule_list):
    # split the data into 2 sets by adding the tag bfrc
    length = len(molecule_list)
    block_type = np.zeros(length, dtype=bool)
    blocksize = length // block_size  # todo: what is block_size?
    for k in range(block_size):
        indrange = slice(k*blocksize, (k+1)*blocksize)
        side = k % 2
        block_type[indrange] = side
    molecule_list_1 = molecule_list[block_type, 1:]
    molecule_list_2 = molecule_list[~block_type, 1:]
    return molecule_list_1, molecule_list_2


def compute_pixel_grid_idx_fs(molecule_list, image_size, pixel_size, show_intermediate_result=False):
    """
    Compute the grid artifact index given the localization data. Using the method of FRC resolution,
    namely split the data into two parts, then compute the FRC value at different frequencies.
    But here we only care about the peaks around the pixel size frequency.

    Args:
        molecule_list (np.ndarray): localization data with [frame, x, y, z, photons...]
        image_size (list): [rows columns] in pixels
        pixel_size (int): physical pixel size, in nm
    """

    # sort the molecule_list by the frame number, namely the first column
    sorted_molecule_list = molecule_list[molecule_list[:, 0].argsort()]

    # split the molecule_list into two parts
    molecule_list_1, molecule_list_2 = make_2_blocks(10, sorted_molecule_list)

    # define the FRC bin size
    super_res_factor = 10
    frc_bin_size = pixel_size/super_res_factor
    sr_image_size = [super_res_factor*image_size[0], super_res_factor*image_size[1]]

    sr_image_1 = get_hist_image(molecule_list_1, frc_bin_size, sr_image_size)
    sr_image_2 = get_hist_image(molecule_list_2, frc_bin_size, sr_image_size)

    sr_image_1_fft = np.fft.fftshift(np.fft.fft2(sr_image_1))
    sr_image_2_fft = np.fft.fftshift(np.fft.fft2(sr_image_2))

    nominator = radial_sum(np.real(sr_image_1_fft*np.conjugate(sr_image_2_fft)))
    denominator = np.sqrt(radial_sum(np.abs(sr_image_1_fft)**2)*radial_sum(np.abs(sr_image_2_fft)**2))

    frc = nominator/denominator

    frequency_max = 1/frc_bin_size/2
    frequency_axis = np.linspace(start=0, stop=frequency_max, num=len(frc))
    resolution_axis = 1/(frequency_axis+np.finfo(float).eps)

    # get the 3 nearest FRC value around the pixel_size frequency
    index = np.abs(resolution_axis - pixel_size).argmin()
    grid_index = np.max(frc[index-1:index+2])

    if show_intermediate_result:
        plt.figure(dpi=400)
        fig, ax_arr = plt.subplots(3, 2, figsize=(6, 9), constrained_layout=True)
        ax_arr[0,0].imshow(sr_image_1,
                           cmap='hot',
                           vmin=np.percentile(sr_image_1, 2),
                           vmax=np.percentile(sr_image_1, 98))
        ax_arr[0,1].imshow(sr_image_2,
                           cmap='hot',
                           vmin=np.percentile(sr_image_2, 2),
                           vmax=np.percentile(sr_image_2, 98))
        ax_arr[1,0].imshow(np.real(sr_image_1_fft),
                           cmap='Greys',
                           vmin=np.percentile(np.real(sr_image_1_fft), 0),
                           vmax=np.percentile(np.real(sr_image_1_fft), 95))
        ax_arr[1,1].imshow(np.real(sr_image_2_fft),
                           cmap='Greys',
                           vmin=np.percentile(np.real(sr_image_2_fft), 0),
                           vmax=np.percentile(np.real(sr_image_2_fft), 95))
        ax_arr[2, 0].plot(frequency_axis, frc)
        ax_arr[2, 0].set_xlabel('Spatial frequency (nm$^{-1}$)')
        ax_arr[2, 0].set_ylabel('FRC')
        ax_arr[2, 0].scatter(frequency_axis[index], grid_index, s=20, c='r')
        ax_arr[2, 0].set_title(f'grid index: {grid_index:.2f}')
        plt.show()
        #plt.savefig("/home/feiyue/liteloc_git/only_local/fft_results/decode_npc_roi_in_roi_frc.svg")

    return grid_index

def calculate_crlb_rmse(loc_model, zstack=25, sampling_num=100):  # for vector psf
    PSF_torch = loc_model.DataGen.VectorPSF
    xemit = torch.tensor(0 * np.ones(zstack))
    yemit = torch.tensor(0 * np.ones(zstack))
    zemit = torch.tensor(1 * np.linspace(-loc_model.params.PSF_model.z_scale, loc_model.params.PSF_model.z_scale, zstack))
    Nphotons = torch.tensor((loc_model.params.Training.photon_range[0] + loc_model.params.Training.photon_range[1]) / 2 * np.ones(zstack)).cuda()
    bg = torch.tensor(
        (loc_model.params.Training.bg - loc_model.params.Camera.baseline) / loc_model.params.Camera.em_gain *
        loc_model.params.Camera.e_per_adu / loc_model.params.Camera.qe * np.ones(zstack)).cuda()

    # calculate crlb and plot
    crlb_xyz, _ = PSF_torch.compute_crlb(xemit, yemit, zemit, Nphotons, bg)
    plt.figure(constrained_layout=True)
    plt.plot(zemit, cpu(crlb_xyz[:, 0]),'b', zemit, cpu(crlb_xyz[:, 1]),'g', zemit, cpu(crlb_xyz[:, 2]),'r')
    plt.legend(('$CRLB_x^{1/2}$', '$CRLB_y^{1/2}$', '$CRLB_z^{1/2}$'), ncol=3, loc='upper center')
    plt.xlim([-loc_model.params.PSF_model.z_scale, loc_model.params.PSF_model.z_scale])
    plt.show()

    # simulate single-molecule data
    xemit = (np.ones(zstack) - 2 * np.random.rand(1)) * loc_model.params.PSF_model.vector_psf.pixelSizeX
    yemit = (np.ones(zstack) - 2 * np.random.rand(1)) * loc_model.params.PSF_model.vector_psf.pixelSizeY
    zemit = torch.tensor(1 * np.linspace(-loc_model.params.PSF_model.z_scale, loc_model.params.PSF_model.z_scale, zstack))
    sampling_data = [[] for i in range(sampling_num*zstack)]
    sampling_gt = [[] for i in range(sampling_num*zstack)]
    frame_count = 0
    for i in tqdm(range(sampling_num)):
        ground_truth = [[] for k in range(zstack)]
        for j in range(zstack):
            frame_count = frame_count + 1
            ground_truth[j] = [frame_count,
                               xemit[j] + loc_model.params.PSF_model.vector_psf.psfSizeX / 2 * loc_model.params.PSF_model.vector_psf.pixelSizeX +
                               loc_model.params.PSF_model.vector_psf.pixelSizeX,
                               yemit[j] + loc_model.params.PSF_model.vector_psf.psfSizeX / 2 * loc_model.params.PSF_model.vector_psf.pixelSizeY +
                               loc_model.params.PSF_model.vector_psf.pixelSizeY,
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
            for i in range(int(np.ceil(sampling_num*zstack/loc_model.params.Training.batch_size))):
                img = sampling_data[i*loc_model.params.Training.batch_size:(i+1)*loc_model.params.Training.batch_size]
                liteloc_molecule_tensor = loc_model.LiteLoc.analyze(img)
                liteloc_molecule_tensor[:, 0] += i * loc_model.params.Training.batch_size
                liteloc_molecule_tensor[:, 1] = liteloc_molecule_tensor[:, 1] * \
                                                loc_model.params.PSF_model.vector_psf.pixelSizeX
                liteloc_molecule_tensor[:, 2] = liteloc_molecule_tensor[:, 2] * \
                                                loc_model.params.PSF_model.vector_psf.pixelSizeY
                liteloc_molecule_tensor[:, 3] = liteloc_molecule_tensor[:, 3] * loc_model.params.PSF_model.z_scale
                liteloc_molecule_tensor[:, 4] = liteloc_molecule_tensor[:, 4] * loc_model.params.Training.photon_range[1]
                liteloc_pred_list[
                liteloc_index_0:liteloc_index_0 + liteloc_molecule_tensor.shape[0]] = liteloc_molecule_tensor
                liteloc_index_0 = liteloc_molecule_tensor.shape[0] + liteloc_index_0

            liteloc_pred = cpu(liteloc_pred_list[:liteloc_index_0]).tolist()
    liteloc_perf_dict, liteloc_matches = loc_model.EvalMetric.limited_matching(sampling_gt, liteloc_pred)

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
    plt.xlim([-loc_model.params.PSF_model.z_scale, loc_model.params.PSF_model.z_scale])
    x_ticks = np.arange(-loc_model.params.PSF_model.z_scale, loc_model.params.PSF_model.z_scale+1, 200)
    plt.xticks(x_ticks)
    plt.ylim(bottom=0)
    plt.tick_params(labelsize=14)
    plt.show()

def recursive_namespace_to_dict(ns):
    return {key: recursive_namespace_to_dict(value) if isinstance(value, recursivenamespace.RecursiveNamespace) else value
            for key, value in vars(ns).items()}
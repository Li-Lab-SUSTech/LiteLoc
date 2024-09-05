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
    blocksize = length // block_size
    for k in range(block_size):
        indrange = slice(k*blocksize, (k+1)*blocksize)
        side = k % 2
        block_type[indrange] = side
    molecule_list_1 = molecule_list[block_type, 1:]
    molecule_list_2 = molecule_list[~block_type, 1:]
    return molecule_list_1, molecule_list_2


def compute_pixel_grid_idx(molecule_list, image_size, pixel_size, show_intermediate_result=False):
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

    return grid_index


if __name__ == '__main__':
    molecule_list = pd.read_csv("../datasets/calculate_grid_index/decode_pos4w5_frame9k_size32_spline.csv")
    # molecule_list = pd.read_csv("../datasets/calculate_grid_index/decode_pos4w5_frame3w_size32_spline.csv")
    # molecule_list = pd.read_csv("../datasets/calculate_grid_index/pos4w5_frame9k_size32.csv")
    molecule_list = np.array(molecule_list)[:, :5]
    image_size = [32, 32]  # pixel number of the raw SMLM data
    pixel_size = 108  # pixel size of the raw SMLM data, nm
    grid_index = compute_pixel_grid_idx(molecule_list, image_size, pixel_size, show_intermediate_result=True)

    # compute_pixel_grid_idx_fy(molecule_list, pixel_size)

    print(f'The grid index of this data is {grid_index}')

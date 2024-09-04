import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftshift
from scipy.signal import find_peaks

molecule_list = pd.read_csv("/home/feiyue/LiteLoc_spline/for_calculate_grid_metric/decode_pos4w5_frame9k_size32_spline.csv")
molecule_list = np.array(molecule_list)[:, 1:5]
image_size = [32, 32]
pixel_size = 108

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

print('end')
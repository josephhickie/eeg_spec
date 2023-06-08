"""
Created on 16/05/2023
@author jdh
"""

import matplotlib
matplotlib.use('TkAgg')

import numpy as np
from numpy.fft import fft, fftfreq, ifft
import matplotlib.pyplot as plt
import scipy
from scipy.integrate import simps
import mne


from src import fieldtrip_to_dict
from src import epoch

filename = '/Users/joseph/Downloads/4007_rest1_ft_attempt2.mat'
sample_rate = 500
data = scipy.io.loadmat(filename)
channel = 'FCZ'
epoch_length_in_seconds = 2


def calculate_band_power(psd, freq_low, freq_high, frequencies, plot=False, title=''):
    frequency_resolution = frequencies[1] - frequencies[0]
    index_delta = np.logical_and(frequencies >= freq_low, frequencies <= freq_high)
    power = simps(psd[index_delta], dx=frequency_resolution)

    if plot:
        plt.figure()
        plt.plot(frequencies, psd)
        plt.fill_between(frequencies, psd, where=(frequencies > freq_low) & (frequencies < freq_high))
        plt.xlim(0, 50)
        plt.title(title)
        plt.show()

    print(f'power is {power:.2f}')
    return power

def relative_band_power(psd, freq_low, freq_high, frequencies, plot=False):
    absolute_power = calculate_band_power(psd, freq_low, freq_high, frequencies, plot)
    total_power = calculate_band_power(psd, frequencies[0], frequencies[-1], frequencies)

    return absolute_power / total_power


# Define the relevant information about the recording setup
ch_names = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'FZ',
            'FCZ', 'CZ', 'PZ', 'FC5', 'FC6', 'FT9', 'FT10', 'CP5', 'CP6', 'P9', 'P10', 'M2', 'HEOR', 'HEOL', 'NOSE',
            'AF3', 'AF4', 'AF7', 'AF8', 'F1', 'F2', 'F5', 'F6', 'FC3', 'FC4', 'FT7', 'FT8', 'C5', 'C6', 'CP3', 'CP4',
            'TP7', 'TP8', 'P1', 'P2', 'P5', 'P6', 'PO3', 'PO4', 'PO7', 'PO8', 'FPZ', 'AFZ', 'CPZ', 'OZ', 'P009', 'P010',
            'VEOG', 'WALL', 'EKG', 'NASION', 'TRIGGER']  # Channel names
ch_types = ['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
            'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eog',
            'eog', 'misc', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
            'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
            'eeg', 'eeg', 'eeg', 'eeg', 'eog', 'misc', 'ecg', 'misc', 'stim']  # Channel types


data_dict = fieldtrip_to_dict(filename, ch_names, sample_rate, ch_types)



# Create an Info object containing the relevant information
info = mne.create_info(ch_names=ch_names, sfreq=sample_rate, ch_types=ch_types)

# Load the EEG data using read_raw_fieldtrip() and pass the info object as an argument
raw = mne.io.read_raw_fieldtrip(filename, info=info)

fcz_data = raw.get_data()[raw.ch_names.index(channel)]

plt.figure()
plt.title('raw_data')
plt.plot(raw.times, fcz_data)
plt.show()


fcz_data = fcz_data[30 * sample_rate: 270 * sample_rate]
fcz_data = fcz_data - np.mean(fcz_data)


plt.figure()
plt.title('with start and end removed, demeaned')
plt.plot(raw.times[30 * sample_rate: 270 * sample_rate], fcz_data)
plt.show()

X = fft(fcz_data)
N = len(X)
n = np.arange(N)
T = N/sample_rate
freq = n/T

x_real = X[:N//2]
f_real = freq[:N//2]

epoched = epoch(fcz_data, sample_rate, epoch_length_in_seconds)
hanning_length = sample_rate * epoch_length_in_seconds

# hanning_window = np.hanning(hanning_length)
hanning_window = np.ones(hanning_length)

ffted_data = []

for epoch in epoched:
    epoch = epoch - epoch.mean()
    hanning_epoch = epoch * hanning_window
    fourier = fft(hanning_epoch)
    ffted_data.append(np.abs(fourier))

ffted_data = np.array(ffted_data)
meaned = ffted_data.mean(axis=0)
# meaned = meaned - meaned.mean()




N = len(meaned)
n = np.arange(N)
T = N/sample_rate
freq = n/T

x_real = meaned[:N//2]
f_real = freq[:N//2]

psd = np.abs(x_real)

plt.figure()
plt.title('fourier transform here')
plt.plot(f_real, psd, 'b')#, markerfmt=" ", basefmt="-b")
plt.xlabel('Hz')
plt.ylabel('power')
plt.xlim(0, 50)
plt.show()


import numpy as np
from scipy.signal import butter, filtfilt, resample, decimate


def butter_lowpass(cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', output='ba')
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=2):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def filter_instance(instance, cutoff_freq, freq):
    instance['x'] = butter_lowpass_filter(instance['x'], cutoff_freq, freq)
    instance['y'] = butter_lowpass_filter(instance['y'], cutoff_freq, freq)
    instance['z'] = butter_lowpass_filter(instance['z'], cutoff_freq, freq)
    return instance


def normalize_instance(instance):
    instance['x'] = instance['x'] - np.mean(instance['x'])
    instance['y'] = instance['y'] - np.mean(instance['y'])
    instance['z'] = instance['z'] - np.mean(instance['z'])
    return instance


def resample_signal(msignal, n_sample):
    return resample(msignal, n_sample)


def decimate_signal(msignal, mdecimate_factor):
    return decimate(msignal, mdecimate_factor)


def normalize_signal(signal):
    return signal - np.mean(signal)


def compute_norm(x, y):
    return np.sqrt(x ** 2 + y ** 2)


class FourPL_Regression:
    def __init__(self, a=0, b=5, c=0.5, d=1):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def compute_confidence(self, x):
        y = self.d + (self.a - self.d) / (1 + (x / self.c) ** self.b)
        return y

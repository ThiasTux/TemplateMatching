from os.path import join
from scipy import stats
import numpy as np

from data_processing.dataset_interface import Dataset
import pandas as pd
import matplotlib.pyplot as plt

from utils.plots import plot_creator


class SHLPreview(Dataset):
    def __init__(self):
        super().__init__()
        self.__dataset_path = join(self.datasets_input_path, "SHL/SHLDataset_preview_v1/")
        self.__users = ['User1', 'User2', 'User3']
        self.__sessions = ['220617', '260617', '270617']
        self.__data_columns = ['time_ms', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'magn_x', 'magn_y',
                               'magn_z', 'quat_w', 'quat_x', 'quat_y', 'quat_z', 'grav_x', 'grav_y', 'grav_z',
                               'lin_acc_x',
                               'lin_acc_y', 'lin_acc_z', 'press', 'alti', 'temp']
        self.__labels_columns = ['time_ms', 'coarse', 'fine', 'road', 'traffic', 'tunnels', 'social', 'food']
        self.__frequency = 100

    def load_isolated_dataset(self):
        dataset_inputpath = join(self.__dataset_path, self.__users[0], self.__sessions[2], 'Hips_Motion.txt')
        data_df = pd.read_csv(dataset_inputpath, sep=" ", names=self.__data_columns)
        data_df['time_ms'] = pd.to_datetime(data_df['time_ms'], unit='ms')
        labels_inputpath = join(self.__dataset_path, self.__users[0], self.__sessions[2], 'Label.txt')
        labels_df = pd.read_csv(labels_inputpath, sep=" ", names=self.__labels_columns)
        labels_df['time_ms'] = pd.to_datetime(labels_df['time_ms'], unit='ms')
        df = pd.merge(data_df, labels_df, on='time_ms')
        window_size_sec = 10
        window_overlap = 0.5
        num_samples_window = int(window_size_sec * self.__frequency)
        samples_step = int(window_size_sec * self.__frequency * window_overlap)
        labels = []
        streams = []
        for i in range(0, len(df) - samples_step, samples_step):
            tmp_value = df['lin_acc_x'].iloc[i:i + num_samples_window].values
            streams.append(self.compute_power_spectrum(tmp_value))
            label = stats.mode(df['coarse'][i: i + num_samples_window])[0][0]
            labels.append(label)
        labels = np.array(labels)
        streams = [streams[t] for t in np.where(labels > 0)[0]]
        streams_labels = labels[np.where(labels > 0)[0]]
        streams_labels_sorted_idx = streams_labels.argsort()
        streams = [streams[i] for i in streams_labels_sorted_idx]
        streams_labels = streams_labels[streams_labels_sorted_idx]
        tmp_streams = [streams[t] for l in np.unique(streams_labels) for t in np.where(streams_labels == l)[0][:100]]
        tmp_labels = [streams_labels[t] for l in np.unique(streams_labels) for t in
                      np.where(streams_labels == l)[0][:100]]
        return tmp_streams, np.array(tmp_labels)

    def load_continuous_dataset(self):
        pass

    def compute_power_spectrum(self, data_frame):
        fourier_transform = np.fft.rfft(data_frame)
        abs_fourier_transform = np.abs(fourier_transform)
        return np.square(abs_fourier_transform)[:100]


def plot_isolated_gestures():
    dataset = SHLPreview()
    streams, labels = dataset.load_isolated_dataset()
    plot_creator.plot_gestures(streams, labels)
    plt.show()


if __name__ == '__main__':
    plot_isolated_gestures()

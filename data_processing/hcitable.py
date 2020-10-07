import glob
from os.path import join

import matplotlib.pyplot as plt
import numpy as np

from data_processing.dataset_interface import Dataset
from template_matching.encode_trajectories import normalize, encode_2d
from utils.filter_data import butter_lowpass_filter
from utils.plots import plot_creator


class HCITable(Dataset):

    @property
    def frequency(self):
        return None

    @property
    def quick_load(self):
        return True

    @property
    def default_classes(self):
        return [i for i in range(9, 35)]

    def __init__(self):
        super().__init__()
        self.users = []

    @property
    def dataset_path(self):
        return join(self.datasets_input_path, "hcitable_release/data/")

    def load_isolated_dataset(self, load_encoded=True):
        if load_encoded:
            return self.load_encoded_isolated_dataset()
        else:
            files = [file for file in glob.glob(self.dataset_path + "*-table*-data.txt")]
            files = sorted(files)
            data = np.loadtxt(files[0])
            table_data = data[:, -4:-2]
            table_data = np.nan_to_num(table_data)
            table_data[:, 0] = butter_lowpass_filter(table_data[:, 0], 5, 200)
            table_data[:, 1] = butter_lowpass_filter(table_data[:, 1], 5, 200)
            labels = data[:, -2]
            labels[np.where(labels < 0)[0]] = 0
            return Dataset.segment_data(table_data, labels)

    def load_encoded_isolated_dataset(self):
        files = [file for file in glob.glob(self.dataset_path + "*-table*-data.txt")]
        files = sorted(files)
        data = np.loadtxt(files[0])
        timestamps = (data[:, 0] * 1000).astype(int)
        table_data = data[:, -4:-2]
        table_data = np.nan_to_num(table_data)
        table_data[:, 0] = butter_lowpass_filter(table_data[:, 0], 5, 200)
        table_data[:, 1] = butter_lowpass_filter(table_data[:, 1], 5, 200)
        labels = data[:, -2]
        labels[np.where(labels < 0)[0]] = 0
        templates, templates_labels = Dataset.segment_data(table_data, labels)
        templates = [templates[t] for t in np.where(templates_labels > 0)[0]]
        templates_labels = templates_labels[np.where(templates_labels > 0)[0]]
        encoded_templates = self.encode_trajectory(templates)
        return encoded_templates, templates_labels

    def load_continuous_dataset(self):
        pass

    def encode_trajectory(self, gestures, frequency=200, sample_freq=50):
        step = int(frequency / sample_freq)
        encoded_gestures = list()
        for gesture_points in gestures:
            gesture_trajectory = list()
            for i in range(0, len(gesture_points) - step, step):
                start_point = gesture_points[i]
                end_point = gesture_points[i + step]
                v_displacement = end_point - start_point
                v_displacement_norm = normalize(v_displacement)
                v_displacement_encoded = encode_2d(v_displacement_norm)
                gesture_trajectory.append(v_displacement_encoded)
            encoded_gestures.append(gesture_trajectory)
        return encoded_gestures


def plot_isolate_gestures(plot=True, save=False):
    dataset = HCITable()
    templates, labels = dataset.load_isolated_dataset()
    templates = [templates[t] for t in np.where(labels > 0)[0]]
    templates = [templates[t] for t in np.where((labels == 17) | (labels == 27))[0]]
    labels = labels[np.where(labels > 0)[0]]
    labels = labels[np.where((labels == 17) | (labels == 27))[0]]
    plot_creator.plot_gestures(templates, labels)
    plt.show()


if __name__ == '__main__':
    plot_isolate_gestures()

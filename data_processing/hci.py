import pickle
from os.path import join

import matplotlib.pyplot as plt
import numpy as np

from data_processing.dataset_interface import Dataset
from utils.filter_data import butter_lowpass_filter, decimate_signal
from utils.plots import plot_creator


class HCIGuided(Dataset):
    @property
    def dataset_path(self):
        return join(self.datasets_input_path, "HCI_FreeHandGestures")

    @property
    def frequency(self):
        return 96

    @property
    def quick_load(self):
        return True

    @property
    def default_classes(self):
        return [49, 50, 51, 52, 53]

    def __init__(self):
        super().__init__()
        self.__users = ["01"]
        self.__default_user = "01"

    def load_data(self):
        datapath = join(self.dataset_path, "usb_hci_guided.csv")
        return np.loadtxt(datapath)

    def load_isolated_dataset(self, sensor_no=51, quick_load=True):
        if quick_load:
            with open(join(self.datasets_input_path, 'WLCSSTraining/datasets/hci/all_data_isolated.pickle'),
                      'rb') as file:
                data = pickle.load(file)
            templates = [d[:, 1] for d in data]
            stream_labels = np.array([d[0, 2] for d in data])
            stream_labels[np.where(stream_labels > 53)[0]] = 0
            return templates, stream_labels
        else:
            data = self.load_data()
            stream_labels = data[:, 0]
            acc_x = decimate_signal(butter_lowpass_filter(data[:, sensor_no], 5, self.frequency), 10)
            acc_y = decimate_signal(butter_lowpass_filter(data[:, sensor_no], 5, self.frequency), 10)
            acc_z = decimate_signal(butter_lowpass_filter(data[:, sensor_no], 5, self.frequency), 10)
            stream_labels = stream_labels[::10]
            stream_labels[np.where(stream_labels > 53)[0]] = 0
            acc_data = np.array([acc_x, acc_y, acc_z]).T
            filtered_data = np.linalg.norm(acc_data, axis=1)
            max_value = 3500
            min_value = 0
            bins = np.arange(min_value, max_value, (max_value - min_value) / 64)
            quantized_data = np.digitize(filtered_data, bins)
            bins = np.arange(0, 64)
            processed_data = np.array([bins[x] for x in quantized_data], dtype=int)
            return Dataset.segment_data(processed_data, stream_labels)

    def load_continuous_dataset(self):
        pass


def plot_isolate_gestures(plot=True, save=False):
    dataset = HCIGuided()
    if plot:
        templates, labels = dataset.load_isolated_dataset()
        labels = np.array(labels)
        templates = [templates[t] for t in np.where(labels > 0)[0]]
        labels = labels[np.where(labels > 0)[0]]
        plot_creator.plot_gestures(templates, labels)
        plt.show()


if __name__ == '__main__':
    plot_isolate_gestures()

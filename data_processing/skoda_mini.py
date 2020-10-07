"""
Skoda mini checkpoint

http://har-dataset.org/doku.php?id=wiki:dataset
"""
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from data_processing.dataset_interface import Dataset
from utils.filter_data import butter_lowpass_filter, decimate_signal
from utils.plots import plot_creator


class SkodaMini(Dataset):

    @property
    def dataset_path(self):
        return join(self.datasets_input_path, "SkodaMiniCP_2015_08/")

    @property
    def frequency(self):
        return 98

    @property
    def quick_load(self):
        return True

    @property
    def default_classes(self):
        return [48, 49, 50, 51, 52, 53, 54, 55, 56, 57]

    def __init__(self):
        super().__init__()
        self.users = ["01"]
        self.default_user = "01"
        self.labels_dict = {32: 'null class', 48: 'write on notepad', 49: 'open hood', 50: 'close hood',
                            51: 'check gaps on the front door', 52: 'open left front door',
                            53: 'close left front door', 54: 'close both left door', 55: 'check trunk gaps',
                            56: 'open and close trunk', 57: 'check steering wheel'}
        self.labels = sorted(list(self.labels_dict.keys()))
        self.sensors = []

    def load_isolated_from_mat(self):
        file = join(self.dataset_path, "dataset_cp_2007_12.mat")
        matdata = loadmat(file)
        left_mat_data = matdata['dataset_left']
        right_mat_data = matdata['dataset_right']
        num_sensors = 30
        num_activities = 10
        num_gestures = 70
        sensor = 16
        sensor_axis = 10
        streams = []
        labels = []
        bins = np.linspace(-1000, 1000, 64)
        for a in range(num_activities):
            # tmp_data_x = [butter_lowpass_filter(t[0]) for t in right_mat_data[0][sensor_axis][0][a][0]]
            # tmp_data_y = [butter_lowpass_filter(t[0]) for t in right_mat_data[0][sensor_axis+1][0][a][0]]
            # tmp_data_z = [butter_lowpass_filter(t[0]) for t in right_mat_data[0][sensor_axis+2][0][a][0]]
            # tmp_data = [np.sqrt(tmp_data_x[i] ** 2 + tmp_data_y[i] ** 2 + tmp_data_z[i] ** 2) for i in range(len(tmp_data_x))]
            tmp_data = [np.digitize(decimate_signal(butter_lowpass_filter(t[0], 5, self.frequency), 10), bins[:-1])
                        for t in right_mat_data[0][sensor_axis][0][a][0]]
            streams += tmp_data
            labels += [self.labels[a + 1] for i in range(len(tmp_data))]
        return streams, np.array(labels)

    def load_isolated_dataset(self, load_templates=True):
        return self.load_isolated_from_mat()

    def load_continuous_dataset(self):
        pass


def plot_isolate_gestures():
    dataset = SkodaMini()
    templates, labels = dataset.load_isolated_from_mat()
    plot_creator.plot_gestures(templates, labels)
    plt.show()


if __name__ == '__main__':
    plot_isolate_gestures()

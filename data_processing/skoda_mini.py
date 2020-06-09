"""
Skoda mini checkpoint

http://har-dataset.org/doku.php?id=wiki:dataset
"""
from os.path import join

from data_processing.dataset_interface import Dataset
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np

from utils.plots import plot_creator
from utils.filter_data import butter_lowpass_filter, decimate_signal


class SkodaMini(Dataset):
    def __init__(self):
        super().__init__()
        self.__dataset_path = join(self.datasets_input_path, "SkodaMiniCP_2015_08/")
        self.__users = ["01"]
        self.__default_user = "01"
        self.__labels_dict = {32: 'null class', 48: 'write on notepad', 49: 'open hood', 50: 'close hood',
                              51: 'check gaps on the front door', 52: 'open left front door',
                              53: 'close left front door', 54: 'close both left door', 55: 'check trunk gaps',
                              56: 'open and close trunk', 57: 'check steering wheel'}
        self.__labels = sorted(list(self.__labels_dict.keys()))
        self.__sensors = []
        self.__frequency = 98

    def load_isolated_from_mat(self):
        file = join(self.__dataset_path, "dataset_cp_2007_12.mat")
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
        for a in range(num_activities):
            # tmp_data_x = [butter_lowpass_filter(t[0]) for t in right_mat_data[0][sensor_axis][0][a][0]]
            # tmp_data_y = [butter_lowpass_filter(t[0]) for t in right_mat_data[0][sensor_axis+1][0][a][0]]
            # tmp_data_z = [butter_lowpass_filter(t[0]) for t in right_mat_data[0][sensor_axis+2][0][a][0]]
            # tmp_data = [np.sqrt(tmp_data_x[i] ** 2 + tmp_data_y[i] ** 2 + tmp_data_z[i] ** 2) for i in range(len(tmp_data_x))]
            tmp_data = [decimate_signal(butter_lowpass_filter(t[0], 5, self.__frequency), 10) for t in
                        right_mat_data[0][sensor_axis][0][a][0]]
            streams += tmp_data
            labels += [self.__labels[a + 1] for i in range(len(tmp_data))]
        return streams, np.array(labels)

    def load_isolated_dataset(self):
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

"""
UTD Multimodal Human Action Dataset

https://personal.utdallas.edu/~kehtar/UTD-MHAD.html
"""
import glob
from os.path import join

from data_processing.dataset_interface import Dataset
from scipy.io import loadmat

from utils.plots import plot_creator
import matplotlib.pyplot as plt

import numpy as np


class UTDMhad(Dataset):

    def __init__(self):
        super().__init__()
        self.__dataset_path = join(self.datasets_input_path, "UTD-MHAD/Inertial/")
        self.__labels = ['a21', 'a23', 'a16', 'a27', 'a10', 'a4', 'a6', 'a26', 'a5', 'a18', 'a12', 'a22', 'a13', 'a9',
                         'a2', 'a1', 'a20', 'a24', 'a15', 'a19', 'a8', 'a25', 'a14', 'a11', 'a17', 'a7', 'a3']
        self.__users = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8']
        self.__default_user = 's2'
        self.__frequency = 50

    def load_isolated_dataset(self, user=None):
        files = [file for file in glob.glob(self.__dataset_path + "*.mat".format(self.default_user))]
        labels = []
        templates = []
        for file in files:
            mat_data = loadmat(file)
            data = mat_data['d_iner']
            label = file.split("/")[-1].split("_")[0]
            labels.append(label)
            tmp_data = np.sqrt(data[:, 0] ** 2 + data[:, 1] ** 2 + data[:, 2] ** 2)
            templates.append(tmp_data)
        return templates, np.array(labels)

    def load_continuous_dataset(self):
        pass


def plot_isolate_gestures():
    dataset = UTDMhad()
    templates, labels = dataset.load_isolated_dataset()
    plot_creator.plot_gestures(templates, labels)
    plt.show()


if __name__ == '__main__':
    plot_isolate_gestures()

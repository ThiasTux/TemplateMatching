"""
Syntethic Dataset created to debug training
"""
import pickle
from os.path import join
import numpy as np
import matplotlib.pyplot as plt

from data_processing.dataset_interface import Dataset
from utils.plots import plot_creator


class Synthetic1(Dataset):

    def __init__(self):
        super().__init__()
        self.__dataset_path = join(self.datasets_input_path, "WLCSSTraining/datasets/synthetic1")

    def load_isolated_dataset(self):
        filepath = join(self.__dataset_path, 'all_data_isolated.pickle')
        with open(filepath, 'rb') as file:
            data = pickle.load(file)
        templates = [d[:, 1] for d in data]
        stream_labels = np.array([d[0, 2] for d in data])
        return templates, stream_labels

    def load_continuous_dataset(self):
        pass


class Synthetic2(Dataset):

    def __init__(self):
        super().__init__()
        self.__dataset_path = join(self.datasets_input_path, "WLCSSTraining/datasets/synthetic2")

    def load_isolated_dataset(self):
        filepath = join(self.__dataset_path, 'all_data_isolated.pickle')
        with open(filepath, 'rb') as file:
            data = pickle.load(file)
        templates = [d[:, 1] for d in data]
        stream_labels = np.array([d[0, 2] for d in data])
        return templates, stream_labels

    def load_continuous_dataset(self):
        pass


class Synthetic3(Dataset):

    def __init__(self):
        super().__init__()
        self.__dataset_path = join(self.datasets_input_path, "WLCSSTraining/datasets/synthetic3")

    def load_isolated_dataset(self):
        filepath = join(self.__dataset_path, 'all_data_isolated.pickle')
        with open(filepath, 'rb') as file:
            data = pickle.load(file)
        templates = [d[:, 1] for d in data]
        stream_labels = np.array([d[0, 2] for d in data])
        return templates, stream_labels

    def load_continuous_dataset(self):
        pass


class Synthetic4(Dataset):

    def __init__(self):
        super().__init__()
        self.__dataset_path = join(self.datasets_input_path, "WLCSSTraining/datasets/synthetic4")

    def load_isolated_dataset(self):
        filepath = join(self.__dataset_path, 'all_data_isolated.pickle')
        with open(filepath, 'rb') as file:
            data = pickle.load(file)
        templates = [d[:, 1] for d in data]
        stream_labels = np.array([d[0, 2] for d in data])
        return templates, stream_labels

    def load_continuous_dataset(self):
        pass


def plot_isolate_gestures():
    dataset = Synthetic4()
    templates, labels = dataset.load_isolated_dataset()
    plot_creator.plot_gestures(templates, labels)
    plt.show()


if __name__ == '__main__':
    plot_isolate_gestures()

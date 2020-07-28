from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample

from data_processing.dataset_interface import Dataset
from utils.plots import plot_creator


class UCR_Dataset(Dataset):
    def __init__(self, dataset_name):
        super().__init__()
        self.__dataset_name = dataset_name
        self.__dataset_path = join(self.datasets_input_path, "UCRArchive_2018", self.__dataset_name)
        self.__templates_length = 315

    def load_isolated_dataset(self):
        data = np.loadtxt(join(self.__dataset_path, "{}_TRAIN.tsv".format(self.__dataset_name)), delimiter="\t")
        labels = data[:, 0]
        resampled_length = int(self.__templates_length / 10)
        # print(np.nanmin(data[:, 1:]))
        # print(np.nanmax(data[:, 1:]))
        bins = np.linspace(-5, 5, 64)
        templates = [np.digitize(resample(t, resampled_length), bins[:-1]) for t in data[:, 1:]]
        return templates, labels

    def load_continuous_dataset(self):
        pass


class AllGestureWiimoteX(UCR_Dataset):
    def __init__(self):
        super().__init__("AllGestureWiimoteX")


class AllGestureWiimoteY(UCR_Dataset):
    def __init__(self):
        super().__init__("AllGestureWiimoteY")


class AllGestureWiimoteZ(UCR_Dataset):
    def __init__(self):
        super().__init__("AllGestureWiimoteZ")


class UWaveGestureLibraryX(UCR_Dataset):
    def __init__(self):
        super().__init__("UWaveGestureLibraryX")


class UWaveGestureLibraryY(UCR_Dataset):
    def __init__(self):
        super().__init__("UWaveGestureLibraryY")


class UWaveGestureLibraryZ(UCR_Dataset):
    def __init__(self):
        super().__init__("UWaveGestureLibraryZ")


def plot_isolated_gestures(plot_null=False):
    dataset = UWaveGestureLibraryX()
    templates, labels = dataset.load_isolated_dataset()
    plot_creator.plot_gestures(templates, labels)
    plt.show()


if __name__ == '__main__':
    plot_isolated_gestures()

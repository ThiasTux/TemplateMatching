"""
Opportunity Dataset

https://archive.ics.uci.edu/ml/datasets/OPPORTUNITY+Activity+Recognition
"""
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from utils.plots import plot_creator

from data_processing.dataset_interface import Dataset


class OpportunityDataset(Dataset):

    def __init__(self):
        super().__init__()
        self.__dataset_path = join(self.datasets_input_path, "OpportunityUCIDataset/dataset/")
        self.__users = ["S1", "S2", "S3", "S4"]
        self.__default_user = "S1"
        self.__frequency = 30

    def load_data(self, user=None):
        if user is None:
            datapath = join(self.__dataset_path, "{}-Drill.dat".format(self.__default_user))
        else:
            datapath = join(self.__dataset_path, "{}-Drill.dat".format(user))
        return np.loadtxt(datapath)

    def load_isolated_dataset(self, sensor_no=67):
        data = self.load_data()
        stream_labels = data[:, 249]
        tmp_data = data[:, sensor_no]
        return Dataset.segment_data(tmp_data, stream_labels)

    def load_continuous_dataset(self):
        pass


def plot_isolate_gestures(sensor_no=67, annotation_column=249):
    dataset = OpportunityDataset()
    data = dataset.load_data()
    stream_labels = data[:, annotation_column]
    tmp_data = data[:, sensor_no]
    templates, labels = Dataset.segment_data(tmp_data, stream_labels)
    labels = np.array(labels)
    templates = [templates[t] for t in np.where(labels > 0)[0]]
    labels = labels[np.where(labels > 0)[0]]
    plot_creator.plot_gestures(templates, labels)
    plt.show()


if __name__ == '__main__':
    plot_isolate_gestures()

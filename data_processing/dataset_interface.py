import glob
import pickle
from abc import abstractmethod, ABC
from os.path import expanduser, join

import numpy as np


class Dataset(ABC):
    datasets_input_path = join(expanduser("~"), "Documents/Datasets")

    def __init__(self):
        pass

    @property
    @abstractmethod
    def dataset_path(self):
        pass

    @property
    @abstractmethod
    def dataset_name(self):
        pass

    @property
    @abstractmethod
    def frequency(self):
        pass

    @property
    @abstractmethod
    def quick_load(self):
        pass

    @property
    @abstractmethod
    def default_classes(self):
        pass

    @abstractmethod
    def load_isolated_dataset(self):
        pass

    @abstractmethod
    def load_continuous_dataset(self):
        pass

    @staticmethod
    def segment_data(data, stream_labels):
        prev_label = None
        start_act = 0
        end_act = -1
        labels = list()
        templates = list()
        for i, label in enumerate(stream_labels):
            if prev_label is None:
                start_act = 0
                prev_label = label
            else:
                if label != prev_label:
                    end_act = i - 1
                    templates.append(data[start_act:end_act])
                    labels.append(prev_label)
                    start_act = i
                    prev_label = label
        return templates, np.array(labels)

    def quick_load_training_dataset(self):
        filepath = [file for file in glob.glob(self.dataset_path + "{}_training_*.pickle".format(self.dataset_name))][0]
        with open(filepath, 'rb') as file:
            templates, streams, stream_labels = pickle.load(file)
        return templates, streams, stream_labels

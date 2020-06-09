from abc import abstractmethod, ABC
import numpy as np

from os.path import expanduser, join


class Dataset(ABC):
    datasets_input_path = join(expanduser("~"), "Documents/Datasets")

    def __init__(self):
        self.__default_user = 's01'
        self.__frequency = None

    @property
    def freq(self):
        return self.__frequency

    @property
    def default_user(self):
        return self.__default_user

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

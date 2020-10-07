"""
Opportunity Dataset

https://archive.ics.uci.edu/ml/datasets/OPPORTUNITY+Activity+Recognition
"""
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
from pyquaternion import Quaternion

from data_processing.dataset_interface import Dataset
from template_matching.encode_trajectories import normalize, encode_3d
from utils.filter_data import butter_lowpass_filter
from utils.plots import plot_creator


class OpportunityDataset(Dataset):

    @property
    def dataset_path(self):
        return join(self.datasets_input_path, "OpportunityUCIDataset/dataset/")

    @property
    def frequency(self):
        return 30

    @property
    def quick_load(self):
        return True

    @property
    def default_classes(self):
        return [406516, 404516, 406520, 404520, 406505, 404505, 406519, 404519, 408512, 407521, 405506]

    def __init__(self):
        super().__init__()
        self.__users = ["S1", "S2", "S3", "S4"]
        self.__default_user = "S1"

    def load_data(self, user=None):
        if user is None:
            datapath = join(self.dataset_path, "{}-Drill.dat".format(self.__default_user))
        else:
            datapath = join(self.dataset_path, "{}-Drill.dat".format(user))
        return np.loadtxt(datapath)

    def load_isolated_dataset(self, sensor_no=67):
        data = self.load_data()
        stream_labels = data[:, 249]
        sensor_data = np.nan_to_num(data[:, sensor_no])
        tmp_data = butter_lowpass_filter(sensor_data, 3, self.frequency)
        bins = np.linspace(-10000, 10000, 128)
        quantized_data = np.digitize(tmp_data, bins[:-1])
        return Dataset.segment_data(quantized_data, stream_labels)

    def load_continuous_dataset(self, sensor_no=67):
        data = self.load_data()
        labels = data[:, 249]
        stream = data[:, sensor_no]
        stream = np.nan_to_num(stream)
        timestamps = data[:, 0]
        return stream, labels, timestamps


class OpportunityDatasetEncoded(Dataset):

    @property
    def dataset_path(self):
        return join(self.datasets_input_path, "OpportunityUCIDataset/dataset/")

    @property
    def frequency(self):
        return 30

    @property
    def quick_load(self):
        return True

    @property
    def default_classes(self):
        return [406516, 404516, 406520, 404520, 406505, 404505, 406519, 404519, 408512, 407521, 405506]

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

    def load_isolated_dataset(self):
        encoded_templates = list()
        templates_labels = np.array([])
        data = self.load_data()
        labels = data[:, -1]
        v_torso = np.array([-6, 0, 0])
        v_others = np.array([3, 0, 0])
        torso_quat = [Quaternion(q) for q in data[:, 46:50]]
        rua_quat = [Quaternion(q) for q in data[:, 59:63]]
        rla_quat = [Quaternion(q) for q in data[:, 72:76]]
        torso_vectors = [q.rotate(v_torso) for q in torso_quat]
        rua_vectors = [rua_quat[i].rotate(v_others) + torso_vectors[i] for i in range(len(rua_quat))]
        rla_vectors = np.array([rla_quat[i].rotate(v_others) + rua_vectors[i] for i in range(len(rla_quat))])
        rla_vectors = np.nan_to_num(rla_vectors)
        rla_vectors[:, 0] = butter_lowpass_filter(rla_vectors[:, 0], 2.5, 30)
        rla_vectors[:, 1] = butter_lowpass_filter(rla_vectors[:, 1], 2.5, 30)
        rla_vectors[:, 2] = butter_lowpass_filter(rla_vectors[:, 2], 2.5, 30)
        templates, tmp_templates_labels = Dataset.segment_data(rla_vectors, labels)
        templates = [templates[t] for t in np.where(tmp_templates_labels > 0)[0]]
        tmp_templates_labels = tmp_templates_labels[np.where(tmp_templates_labels > 0)[0]]
        encoded_templates += self.encode_isolated_trajectories(templates)
        templates_labels = np.append(templates_labels, tmp_templates_labels)
        return encoded_templates, templates_labels

    def load_continuous_dataset(self, sensor_no=67):
        data = self.load_data()
        labels = data[:, 249]
        stream = data[:, sensor_no]
        stream = np.nan_to_num(stream)
        timestamps = data[:, 0]
        return stream, labels, timestamps

    def encode_isolated_trajectories(self, gestures, frequency=None, sample_freq=10):
        if frequency is None:
            frequency = self.__frequency
        step = int(frequency / sample_freq)
        encoded_gestures = list()
        for gesture_points in gestures:
            gesture_trajectory = list()
            for i in range(0, len(gesture_points) - step, step):
                start_point = gesture_points[i]
                end_point = gesture_points[i + step]
                v_displacement = end_point - start_point
                v_displacement_norm = normalize(v_displacement)
                v_displacement_encoded = encode_3d(v_displacement_norm)
                gesture_trajectory.append(v_displacement_encoded)
            encoded_gestures.append(gesture_trajectory)
        return encoded_gestures


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


def plot_encoded_gestures():
    dataset = OpportunityDatasetEncoded()
    templates, labels = dataset.load_isolated_dataset()
    templates = [templates[t] for t in np.where(labels > 0)[0]]
    labels = labels[np.where(labels > 0)[0]]
    plot_creator.plot_gestures(templates, labels)
    plt.show()


def plot_continuous_data():
    dataset = OpportunityDataset()
    stream, labels, timestamps = dataset.load_continuous_dataset()
    plot_creator.plot_continuous_data(stream, labels, timestamps)
    plt.show()


if __name__ == '__main__':
    plot_encoded_gestures()
    # plot_continuous_data()

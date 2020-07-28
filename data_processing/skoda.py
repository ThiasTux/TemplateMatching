"""
Skoda dataset
"""
import glob
import pickle
from os.path import join
import numpy as np
import matplotlib.pyplot as plt

from data_processing.dataset_interface import Dataset
from mpl_toolkits.mplot3d import Axes3D

from template_matching.encode_trajectories import normalize, encode_3d
from utils.filter_data import butter_lowpass_filter
from utils.plots import plot_creator
from pyquaternion import Quaternion


class Skoda(Dataset):
    def __init__(self):
        super().__init__()
        self.__dataset_path = join(self.datasets_input_path, "SkodaDataset/processed_data/")
        self.__users = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        self.__labels_dict = {0: 'null', 3001: 'open_hood', 3002: 'close_hood', 3003: 'open_trunk', 3004: 'check_trunk',
                              3005: 'close_trunk', 3006: 'fuel_lid', 3007: 'open_left_door', 3008: 'close_left_door',
                              3009: 'open_right_door', 3010: 'close_right_door', 3011: 'open_two_doors',
                              3012: 'close_two_doors', 3013: 'mirror', 3014: 'check_trunk_gaps',
                              3015: 'lock_check_left',
                              3016: 'lock_check_right', 3017: 'check_hood_gaps', 3018: 'open_swl', 3019: 'close_swl',
                              3020: 'writing'}
        self.__labels = sorted(list(self.__labels_dict.keys()))
        self.__default_user = 'H'
        self.__default_session_id = '1'
        self.__frequency = 50

    def load_isolated_dataset(self, quick_load=False, load_encoded=True):
        if quick_load:
            quick_load_filepath = join(self.datasets_input_path,
                                       'WLCSSTraining/datasets/skoda/all_data_isolated.pickle')
            with open(quick_load_filepath, 'rb') as file:
                data = pickle.load(file)
            templates = [d[:, 1] for d in data]
            stream_labels = np.array([d[0, 2] for d in data])
            return templates, stream_labels
        else:
            if load_encoded:
                return self.load_encode_dataset()

    def load_continuous_dataset(self, use_torso=False):
        file = join(self.__dataset_path, "subject{}_{}.txt".format(self.__default_user, self.__default_session_id))
        data = np.loadtxt(file)
        timestamps = data[:, 0]
        labels = data[:, -1]
        v_torso = np.array([-6, 0, 0])
        v_others = np.array([3, 0, 0])
        v_hand = np.array([1, 0, 0])
        torso_quat = [Quaternion(q) for q in data[:, 13:17]]
        rua_quat = [Quaternion(q) for q in data[:, 29:33]]
        rla_quat = [Quaternion(q) for q in data[:, 45:49]]
        rha_quat = [Quaternion(q) for q in data[:, 61:65]]
        if use_torso:
            torso_vectors = [q.rotate(v_torso) for q in torso_quat]
            rua_vectors = [rua_quat[i].rotate(v_others) + torso_vectors[i] for i in range(len(rua_quat))]
        else:
            rua_vectors = [rua_quat[i].rotate(v_others) for i in range(len(rua_quat))]
        rla_vectors = [rla_quat[i].rotate(v_others) + rua_vectors[i] for i in range(len(rla_quat))]
        rha_vectors = np.array([rha_quat[i].rotate(v_hand) + rla_vectors[i] for i in range(len(rha_quat))])
        rha_vectors[:, 0] = butter_lowpass_filter(rha_vectors[:, 0], 2.5, 30)
        rha_vectors[:, 1] = butter_lowpass_filter(rha_vectors[:, 1], 2.5, 30)
        rha_vectors[:, 2] = butter_lowpass_filter(rha_vectors[:, 2], 2.5, 30)
        encoded_stream, labels, timestamps = self.encode_continuous_trajectory(rha_vectors, labels, timestamps)
        return encoded_stream, labels, timestamps

    def load_encode_dataset(self, use_torso=False):
        files = [file for file in glob.glob(join(self.__dataset_path, "subject{}_*.txt".format(self.__default_user)))]
        encoded_templates = list()
        templates_labels = np.array([])
        for file in files:
            data = np.loadtxt(file)
            labels = data[:, -1]
            v_torso = np.array([-6, 0, 0])
            v_others = np.array([3, 0, 0])
            v_hand = np.array([1, 0, 0])
            torso_quat = [Quaternion(q) for q in data[:, 13:17]]
            rua_quat = [Quaternion(q) for q in data[:, 29:33]]
            rla_quat = [Quaternion(q) for q in data[:, 45:49]]
            rha_quat = [Quaternion(q) for q in data[:, 61:65]]
            if use_torso:
                torso_vectors = [q.rotate(v_torso) for q in torso_quat]
                rua_vectors = [rua_quat[i].rotate(v_others) + torso_vectors[i] for i in range(len(rua_quat))]
            else:
                rua_vectors = [rua_quat[i].rotate(v_others) for i in range(len(rua_quat))]
            rla_vectors = [rla_quat[i].rotate(v_others) + rua_vectors[i] for i in range(len(rla_quat))]
            rha_vectors = np.array([rha_quat[i].rotate(v_hand) + rla_vectors[i] for i in range(len(rha_quat))])
            rha_vectors[:, 0] = butter_lowpass_filter(rha_vectors[:, 0], 2.5, 30)
            rha_vectors[:, 1] = butter_lowpass_filter(rha_vectors[:, 1], 2.5, 30)
            rha_vectors[:, 2] = butter_lowpass_filter(rha_vectors[:, 2], 2.5, 30)
            templates, tmp_templates_labels = Dataset.segment_data(rha_vectors, labels)
            templates = [templates[t] for t in np.where(tmp_templates_labels > 0)[0]]
            tmp_templates_labels = tmp_templates_labels[np.where(tmp_templates_labels > 0)[0]]
            encoded_templates += self.encode_isolated_trajectories(templates)
            templates_labels = np.append(templates_labels, tmp_templates_labels)
        return encoded_templates, templates_labels

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

    def encode_continuous_trajectory(self, stream, stream_labels=None, stream_timestamps=None, frequency=None,
                                     sample_freq=10):
        if frequency is None:
            frequency = self.__frequency
        step = int(frequency / sample_freq)
        trajectory = list()
        gesture_trajectory = list()
        for i in range(0, len(stream) - step, step):
            start_point = stream[i]
            end_point = stream[i + step]
            v_displacement = end_point - start_point
            v_displacement_norm = normalize(v_displacement)
            v_displacement_encoded = encode_3d(v_displacement_norm)
            gesture_trajectory.append(v_displacement_encoded)
        if stream_labels is not None:
            stream_labels = [stream_labels[i] for i in range(0, len(stream) - step, step)]
            stream_timestamps = [stream_timestamps[i] for i in range(0, len(stream) - step, step)]
            return np.array(gesture_trajectory), np.array(stream_labels), np.array(stream_timestamps)
        else:
            return np.array(gesture_trajectory)

    def plot_live_gestures(self):
        file = join(self.__dataset_path, "subject{}_{}.txt".format(self.__default_user, self.__default_session_id))
        data = np.loadtxt(file)
        timestamps = data[:, 0]
        labels = data[:, -1]
        v_torso = np.array([-6, 0, 0])
        v_others = np.array([3, 0, 0])
        v_hand = np.array([1, 0, 0])
        torso_quat = [Quaternion(q) for q in data[:, 13:17]]
        rua_quat = [Quaternion(q) for q in data[:, 29:33]]
        rla_quat = [Quaternion(q) for q in data[:, 45:49]]
        rha_quat = [Quaternion(q) for q in data[:, 61:65]]
        torso_vectors = [q.rotate(v_torso) for q in torso_quat]
        rua_vectors = [rua_quat[i].rotate(v_others) + torso_vectors[i] for i in range(len(rua_quat))]
        rla_vectors = [rla_quat[i].rotate(v_others) + rua_vectors[i] for i in range(len(rla_quat))]
        rha_vectors = [rha_quat[i].rotate(v_hand) + rla_vectors[i] for i in range(len(rha_quat))]
        fig = plt.figure()
        plt.ion()
        vector_plot = fig.add_subplot(111, projection="3d")
        vector_plot.set_xlim([-7, 7])
        vector_plot.set_ylim([-7, 7])
        vector_plot.set_zlim([-7, 7])
        vector_plot.set_xlabel('X')
        vector_plot.set_ylabel('Y')
        vector_plot.set_zlabel('Z')
        i = 0
        v_torso_plot = None
        while True:
            if v_torso_plot:
                # vector_plot.collections.remove(v_torso_plot)
                v_torso_plot.pop(0).remove()
                v_rua_plot.pop(0).remove()
                v_rla_plot.pop(0).remove()
                v_rha_plot.pop(0).remove()
                # vector_plot.collections.remove(v_rua_plot)
                # vector_plot.collections.remove(v_rla_plot)
                # vector_plot.collections.remove(v_rha_plot)
            v_torso_plot = vector_plot.plot([0, torso_vectors[i][0]],
                                            [0, torso_vectors[i][1]],
                                            [0, torso_vectors[i][2]],
                                            color='k')
            v_rua_plot = vector_plot.plot([torso_vectors[i][0], rua_vectors[i][0]],
                                          [torso_vectors[i][1], rua_vectors[i][1]],
                                          [torso_vectors[i][2], rua_vectors[i][2]], color='r')
            v_rla_plot = vector_plot.plot([rua_vectors[i][0], rla_vectors[i][0]],
                                          [rua_vectors[i][1], rla_vectors[i][1]],
                                          [rua_vectors[i][2], rla_vectors[i][2]], color='g')
            v_rha_plot = vector_plot.plot([rla_vectors[i][0], rha_vectors[i][0]],
                                          [rla_vectors[i][1], rha_vectors[i][1]],
                                          [rla_vectors[i][2], rha_vectors[i][2]], color='b')
            print(labels[i])
            i += 1
            plt.pause(1 / self.__frequency)


def plot_isolate_gestures():
    dataset = Skoda()
    templates, labels = dataset.load_isolated_dataset()
    plot_creator.plot_gestures(templates, labels)
    plt.show()


def plot_continuous_data():
    dataset = Skoda()
    stream, labels, timestamps = dataset.load_continuous_dataset()
    plot_creator.plot_continuous_data(stream, labels, timestamps)
    plt.show()


if __name__ == '__main__':
    plot_isolate_gestures()
    # plot_continuous_data()

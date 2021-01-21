import glob
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
from pyquaternion import Quaternion

from data_processing.dataset_interface import Dataset
from template_matching.encode_trajectories import normalize, encode_3d
from utils.filter_data import butter_lowpass_filter, decimate_signal
from utils.plots import plot_creator


class BeachVolleyball(Dataset):
    @property
    def dataset_path(self):
        return join(self.datasets_input_path, "BeachVolleyball/")

    @property
    def dataset_name(self):
        return 'beachvolleyball'

    @property
    def frequency(self):
        return 500

    @property
    def quick_load(self):
        return True

    @property
    def default_classes(self):
        return [1001, 1002, 1003, 1004]

    def load_isolated_dataset(self):
        templates = []
        labels = []
        datapaths = [file for file in glob.glob(self.dataset_path + "./**/user*/labelled_data.txt", recursive=True)]
        datapaths = sorted(datapaths)
        for d in datapaths[3:5]:
            with open(d) as f:
                data = np.array([line.strip().split() for line in f], float)
            # data = np.loadtxt(datapaths[0])
            tmp_labels = data[:, -1]
            acc_data = data[:, 43:46]
            acc_x = decimate_signal(butter_lowpass_filter(acc_data[:, 0], 10, 500), 5)
            acc_y = decimate_signal(butter_lowpass_filter(acc_data[:, 1], 10, 500), 5)
            acc_z = decimate_signal(butter_lowpass_filter(acc_data[:, 2], 10, 500), 5)
            acc_data = np.array([acc_x, acc_y, acc_z]).T
            # acc_y = data[:, 41]
            # acc_z = data[:, 42]
            tmp_labels_idx = tmp_labels[:-1] - tmp_labels[1:]
            idx_last_useful_label = np.where(tmp_labels_idx == -1004)[0][-1]
            useful_data = acc_data[:idx_last_useful_label]
            useful_labels = tmp_labels[:idx_last_useful_label:5]
            norm_data = np.linalg.norm(useful_data, axis=1)
            print(max(norm_data))
            print(min(norm_data))
            bins = np.linspace(0, 42000, 128)
            quantized_data = np.digitize(norm_data, bins[:-1])
            tmp_templates, tmp_templates_labels = Dataset.segment_data(quantized_data, useful_labels)
            templates += [tmp_templates[t] for t in np.where(tmp_templates_labels > 0)[0]]
            labels = np.append(labels, tmp_templates_labels[np.where(tmp_templates_labels > 0)[0]])
        return templates, labels

    def load_continuous_dataset(self):
        pass


class BeachVolleyballEncoded(Dataset):
    @property
    def dataset_path(self):
        return join(self.datasets_input_path, "BeachVolleyball/")

    @property
    def dataset_name(self):
        return 'beachvolleyball_encoded'

    @property
    def frequency(self):
        return 500

    @property
    def quick_load(self):
        return True

    @property
    def default_classes(self):
        return [1001, 1002, 1003, 1004]

    def load_isolated_dataset(self):
        encoded_templates = list()
        templates_labels = np.array([])
        datapaths = [file for file in glob.glob(self.dataset_path + "./**/user*/labelled_data.txt", recursive=True)]
        datapaths = sorted(datapaths)
        for d in datapaths[1:2]:
            with open(datapaths[3]) as f:
                data = np.array([line.strip().split() for line in f], float)
            # data = np.loadtxt(datapaths[0])
            labels = data[:, -1]
            tmp_labels = labels[:-1] - labels[1:]
            idx_last_useful_label = np.where(tmp_labels == -1004)[0][-1]
            data = data[:idx_last_useful_label]
            labels = labels[:idx_last_useful_label]
            v_torso = np.array([0, -6, 0])
            v_limbs = np.array([0, 3, 0])
            v_hand = np.array([0, 1, 0])
            torso_quat = [Quaternion(q) for q in data[:, 10:14]]
            rua_quat = [Quaternion(q) for q in data[:, 23:27]]
            rla_quat = [Quaternion(q) for q in data[:, 36:40]]
            rha_quat = [Quaternion(q) for q in data[:, 49:53]]
            torso_vectors = [q.rotate(v_torso) for q in torso_quat]
            rua_vectors = [rua_quat[i].rotate(v_limbs) + torso_vectors[i] for i in range(len(rua_quat))]
            rla_vectors = np.array([rla_quat[i].rotate(v_limbs) + rua_vectors[i] for i in range(len(rla_quat))])
            rha_vectors = np.array([rha_quat[i].rotate(v_hand) for i in range(len(rha_quat))])
            rha_vectors[:, 0] = butter_lowpass_filter(rha_vectors[:, 0], 10, 500)
            rha_vectors[:, 1] = butter_lowpass_filter(rha_vectors[:, 1], 10, 500)
            rha_vectors[:, 2] = butter_lowpass_filter(rha_vectors[:, 2], 10, 500)
            templates, tmp_templates_labels = Dataset.segment_data(rha_vectors, labels)
            tmp_templates = [templates[t] for t in np.where(tmp_templates_labels > 0)[0]]
            tmp_templates_labels = tmp_templates_labels[np.where(tmp_templates_labels > 0)[0]]
            encoded_templates += self.encode_isolated_trajectories(tmp_templates)
            # encoded_templates += tmp_templates
            templates_labels = np.append(templates_labels, tmp_templates_labels)
        return encoded_templates, templates_labels

    def load_continuous_dataset(self):
        datapaths = [file for file in glob.glob(self.dataset_path + "./**/user*/labelled_data.txt", recursive=True)]
        datapaths = sorted(datapaths)
        print(datapaths[3])
        with open(datapaths[3]) as f:
            data = np.array([line.strip().split() for line in f], float)
        labels = data[:, -1]

    def encode_isolated_trajectories(self, gestures, frequency=None, sample_freq=50):
        if frequency is None:
            frequency = self.frequency
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


def plot_isolated_encoded_gestures():
    dataset = BeachVolleyballEncoded()
    templates, labels = dataset.load_isolated_dataset()
    plot_creator.plot_3d_gestures(templates, labels)
    plt.show()


def plot_isolated_gestures():
    dataset = BeachVolleyball()
    templates, labels = dataset.load_isolated_dataset()
    plot_creator.plot_gestures(templates, labels)
    plt.show()


if __name__ == '__main__':
    plot_isolated_encoded_gestures()
    # plot_isolated_gestures()

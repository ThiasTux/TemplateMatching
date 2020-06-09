import glob
import os
import struct
from os.path import join
from subprocess import call, check_output
from data_processing.dataset_interface import Dataset
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

from utils.plots import plot_creator

INPUT_FOLDER = "/home/mathias/Documents/Datasets/Wetlab/"

wetlab_labels = ['cutting', 'inverting', 'peeling', 'pestling', 'pipetting', 'pour catalysator', 'pouring', 'stirring',
                 'transfer']
wetlab_labels_dict = {wtl: 2000 + i for i, wtl in enumerate(wetlab_labels)}
wetlab_labels_dict_type = {'Default': 1001, 'action': 1002}


class WetlabDataset(Dataset):

    def __init__(self):
        self.__dataset_path = join(self.datasets_input_path, "Wetlab")
        self.__users = ['101', '102', '104', '105', '107', '108', '109', '110', '112', '113', '114', '117', '118',
                        '125', '128', '131', '134', '135', '137', '139', '141', '142']
        self.__default_user = '102'
        self.__frequency = 50

    def load_data(self, user=None):
        if user is None:
            datapath = join(self.__dataset_path, "{}_acc_labelled.csv".format(self.__default_user))
        else:
            datapath = join(self.__dataset_path, "{}_acc_labelled.csv".format(user))
        df = pd.read_csv(datapath, header=None)
        df.drop(df.columns[-2], axis=1, inplace=True)
        return df.values

    def load_isolated_dataset(self, sensor_no=1):
        data = self.load_data()
        stream_labels = data.iloc[:, -1].values
        tmp_data = data[:, sensor_no]
        return Dataset.segment_data(tmp_data, stream_labels)

    def load_continuous_dataset(self):
        pass


def mkv_list_datafile():
    """
    List all mkv files in INPUT_FOLDER
    :return:
    """
    return [file for file in glob.glob(INPUT_FOLDER + "*.mkv")]


def mkv_extract_data():
    """
    Extract accelerometer data from mkv files.
    :return:
    """
    devnull = open(os.devnull, 'w')
    files = sorted(mkv_list_datafile())
    for file in files:
        bin_file_name = file.replace(".mkv", "_acc.bin")
        extract_bin_command = "ffmpeg -i {} -f f32le - > {}".format(file, bin_file_name)
        call(extract_bin_command, shell=True, stderr=devnull)
        with open(bin_file_name, 'rb') as bin_file:
            data = list()
            while True:
                b = bin_file.read(12)
                if not b:
                    break
                x = struct.unpack("3f", b)
                # y = struct.unpack("3f", data_bin[i + 4:i + 8])
                # z = struct.unpack("3f", data_bin[i + 8:i + 12])
                data.append(list(x))
        output_file = file.replace(".mkv", "_acc.txt")
        np.savetxt(output_file, np.array(data), fmt="%8.7f,%8.7f,%8.7f")
        remove_command = "rm {}".format(bin_file_name)
        call(remove_command, shell=True, stderr=devnull)


def mkv_extract_labels():
    """
    Extract labels from mkv files.
    :return:
    """
    files = sorted(mkv_list_datafile())
    for file in files:
        print(file)
        data_string = check_output(["ffmpeg", "-i", file, "-f", "ass", "-"]).decode("utf-8")
        lines = data_string.split("\n")
        start_events_line = lines.index("[Events]")
        labels_data = list()
        for line in lines[start_events_line + 2:-1]:
            line_values = line.split(",")
            start_time = line_values[1].strip()
            end_time = line_values[2].strip()
            act_type = line_values[3].strip()
            act = line_values[-1].strip()
            labels_data.append([start_time, end_time, act_type, act])
        labels_data = np.array(labels_data)
        np.savetxt(file.replace(".mkv", "_labels.csv"), labels_data, fmt='%s,%s,%s,%s')
    pass


def merge_labels():
    """
    Merge labels to accelerometer data. The labels are formatted as:
    start_time,end_time,act_type,activity
    :return:
    """
    files = sorted(mkv_list_datafile())
    for file in files:
        acc_file = file.replace(".mkv", "_acc.txt")
        label_file = file.replace(".mkv", "_labels.csv")
        acc_data = np.loadtxt(acc_file, delimiter=",")
        final_data = np.zeros((len(acc_data), 4))
        final_data[:, 1:4] = acc_data
        final_data[:, 0] = np.arange(0, len(acc_data) * (1000 / 50), 1000 / 50)
        act_1_labels = ['' for _ in range(len(acc_data))]
        act_2_labels = [0 for _ in range(len(acc_data))]
        with open(label_file, 'r') as labels:
            for line in labels.readlines():
                split_values = line.split(",")
                start_time = split_values[0]
                end_time = split_values[1]
                act_type = split_values[2].strip()
                act = split_values[3].strip()
                start_time_ms = convert_time(start_time)
                end_time_ms = convert_time(end_time)
                start_pos = np.where(final_data[:, 0] > start_time_ms)[0][0]
                end_pos = np.where(final_data[:, 0] > end_time_ms)[0][0]
                if act_type == 'Default':
                    for i in range(start_pos, end_pos):
                        act_1_labels[i] = act
                else:
                    for i in range(start_pos, end_pos):
                        act_2_labels[i] = wetlab_labels_dict[act]
        df = pd.DataFrame(data=final_data, columns=["timestamp", "acc_x", "acc_y", "acc_z"])
        df['act_glasses'] = np.array(act_1_labels)
        df['action'] = np.array(act_2_labels)
        df.to_csv(file.replace(".mkv", "_acc_labelled.csv"), header=None, index=None, sep=',')
        print(file)


def check_labels():
    files = mkv_list_datafile()
    for file in files:
        df = pd.read_csv(file.replace(".mkv", "_acc_labelled.csv"),
                         names=['timestamps', 'acc_x', 'acc_y', 'acc_z', 'act_1', 'act_2'])
        df['act_1'] = df['act_1'].astype('str')
        print(sorted(df['act_1'].unique()))


def convert_time(timestamp_sec):
    """
    Convert timestamp from H:MM:SS.mm format to milliseconds
    :param timestamp_sec: string, timestamp
    :return: timestamp in milliseconds
    """
    ds = int(timestamp_sec[-2:]) * 10
    sec = int(timestamp_sec[-5:-3]) * 1000
    min = int(timestamp_sec[-8:-6]) * 60 * 1000
    hour = int(timestamp_sec[0]) * 60 * 60 * 1000
    return hour + min + sec + ds


def plot_isolate_gestures():
    dataset = WetlabDataset()
    data = dataset.load_data()
    timestamps = data[:, 0]
    stream_labels = data[:, -1]
    tmp_data = np.sqrt(data[:, 1] ** 2 + data[:, 2] ** 2 + data[:, 3] ** 2)
    # tmp_data = data[:, 2]
    templates, labels = Dataset.segment_data(tmp_data, stream_labels)
    labels = np.array(labels)
    templates = [templates[t] for t in np.where(labels > 0)[0]]
    labels = labels[np.where(labels > 0)[0]]
    plot_creator.plot_gestures(templates, labels)
    # subplt_1.plot(timestamps, data[:, sensor_no], linewidth=0.5)
    # subplt_2.plot(timestamps, data[:, 249], '.')
    plt.show()


if __name__ == '__main__':
    # mkv_extract_data()
    # mkv_extract_labels()
    # merge_labels()
    # check_labels()
    plot_isolate_gestures()

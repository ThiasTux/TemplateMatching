import glob
import math
import os
import pickle
from os.path import expanduser
from os.path import join

import matlab.engine
import numpy as np
from pyquaternion import Quaternion
from scipy.stats import norm, uniform

from utils import codebook_builder as cc, filter_data as fd

OUTPUT_FOLDER = "outputs/datasets/"

OPPORTUNITY_FOLDER = join(expanduser("~"), "Documents/Datasets/OpportunityUCIDataset/dataset")
OPPORTUNITY_DOWNSAMPLING_FACTOR = 3

UNILEVER_DRINKING = join(expanduser("~"), "Documents/Datasets/CupLogger/Training/z_training_data.csv")

SKODA_FOLDER = join(expanduser("~"), "Documents/Datasets/SkodaDataset/processed_data/")
SKODA_OLD_FILE = join(expanduser("~"),
                      "Developer/Projects/PyCharmProjects/GestureRecognitionToolchain/datasets/skoda_enc_data_27.txt")
SKODA_USER_DICT = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7}
SKODA_COLS = [np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 14, 15, 16]) + (i * 16) for i in range(7)]
SKODA_COLS = [item for sublist in SKODA_COLS for item in sublist]
SKODA_DATA_IDX = {"torso": [13, 14, 15, 16],
                  "uarm": [29, 30, 31, 32],
                  "larm": [45, 46, 47, 48],
                  "hand": [61, 62, 63, 64]}
SKODA_RESCALING_FACTOR = 10
V = np.array([0., 1., 0.])
SKODA_TIME_INTERVAL = 0.5
SKODA_MIN_DISPLACEMENT = 0.03


def extract_isolated_opportunity(quantize_data=True, sensor=29):
    files = [file for file in glob.glob(OPPORTUNITY_FOLDER + "/*-Drill.dat") if
             os.stat(file).st_size != 0]
    eng = matlab.engine.start_matlab()
    dataset_name = "opportunity"
    all_data = list()
    for file in files:
        user_no = file.split("/")[-1].replace("-Drill.dat", "").replace("S", "0")
        data = np.loadtxt(file, dtype=float)
        if quantize_data:
            tmp_data = np.empty((int(math.ceil(len(data) / OPPORTUNITY_DOWNSAMPLING_FACTOR)), 3), dtype=int)
            used_data = data[:, sensor]
            np.nan_to_num(used_data, False)
            cutoff_freq = 10
            freq = 30
            filtered_data = fd.butter_lowpass_filter(used_data, cutoff_freq, freq)
            processed_data = fd.decimate_signal(filtered_data, OPPORTUNITY_DOWNSAMPLING_FACTOR)
            max_value = 2000
            min_value = -2000
            bins = np.arange(min_value, max_value, (max_value - min_value) / 127)
            digitized_data = np.digitize(processed_data, bins)
            bins = np.arange(-64, 64)
            quantized_data = np.array([bins[x] for x in digitized_data], dtype=int)
            tmp_data[:, 0] = quantized_data
            tmp_data[:, 1] = data[::OPPORTUNITY_DOWNSAMPLING_FACTOR, -2]
            tmp_data[:, 2] = data[::OPPORTUNITY_DOWNSAMPLING_FACTOR, -1]
            data = tmp_data
        labels = data[:, -1]
        [i1, i2, i3] = eng.dtcFindInstancesFromLabelStream(matlab.double(list(labels)), nargout=3)
        for i, c in enumerate(np.unique(labels)):
            m_range = i3[i]['range']
            num_inst = len(m_range)
            for k in range(num_inst):
                extracted_data = data[int(i3[i]['range'][k][0] - 1):int(i3[i]['range'][k][1]),
                                 0 if quantize_data else 37:102]
                if extracted_data.size != 0:
                    extracted_data_time = data[int(i3[i]['range'][k][0] - 1):int(i3[i]['range'][k][1]), 0]
                    tmp_data = np.empty((extracted_data.shape[0], extracted_data.shape[1] + 3))
                    tmp_data[:, 0] = extracted_data_time
                    tmp_data[:, 1:-2] = extracted_data
                    tmp_data[:, -2] = np.array([c for i in range(extracted_data.shape[0])])
                    tmp_data[:, -1] = np.array([user_no for i in range(extracted_data.shape[0])])
                    all_data.append(tmp_data)
    eng.quit()
    if quantize_data:
        with open(join(OUTPUT_FOLDER, dataset_name, "all_quant_data_isolated.pickle"), "wb") as output_file:
            pickle.dump(all_data, output_file)
    else:
        with open(join(OUTPUT_FOLDER, dataset_name, "all_data_isolated.pickle"), "wb") as output_file:
            pickle.dump(all_data, output_file)


# def quantize_opportunity(sensors=28):
#     data = pickle.load(open("outputs/datasets/opportunity/all_data_isolated.pickle", "rb"))
#     new_data = list()
#     dataset_name = "opportunity"
#     for d in data:
#         tmp_data = np.empty((int(math.ceil(len(d) / OPPORTUNITY_DOWNSAMPLING_FACTOR)), 3), dtype=int)
#         used_data = d[:, sensors]
#         np.nan_to_num(used_data, False)
#         cutoff_freq = 10
#         freq = 30
#         filtered_data = fd.butter_lowpass_filter(used_data, cutoff_freq, freq)
#         processed_data = fd.decimate_signal(filtered_data, OPPORTUNITY_DOWNSAMPLING_FACTOR)
#         max_value = 2000
#         min_value = -2000
#         bins = np.arange(min_value, max_value, (max_value - min_value) / 127)
#         digitized_data = np.digitize(processed_data, bins)
#         bins = np.arange(-64, 64)
#         quantized_data = np.array([bins[x] for x in digitized_data], dtype=int)
#         tmp_data[:, 0] = quantized_data
#         tmp_data[:, 1] = d[::OPPORTUNITY_DOWNSAMPLING_FACTOR, -2]
#         tmp_data[:, 2] = d[::OPPORTUNITY_DOWNSAMPLING_FACTOR, -1]
#         new_data.append(tmp_data)
#     with open(join(OUTPUT_FOLDER, dataset_name, "quant_rla_data_isolated.pickle"), "wb") as output_file:
#         pickle.dump(new_data, output_file)


def extract_continuous_opportunity():
    files = [file for file in glob.glob(OPPORTUNITY_FOLDER + "/*-Drill.dat") if
             os.stat(file).st_size != 0]
    dataset_name = "opportunity"
    for file in files:
        user_no = file.split("/")[-1].replace("-Drill.dat", "").replace("S", "0")
        data = np.loadtxt(file, dtype=float)
        extracted_data = data[:, 37:102]
        extracted_data_time = data[:, 0]
        extracted_data_labels = data[:, -1]
        tmp_data = np.empty((extracted_data.shape[0], extracted_data.shape[1] + 3))
        tmp_data[:, 0] = extracted_data_time
        tmp_data[:, 1:-2] = extracted_data
        tmp_data[:, -2] = extracted_data_labels
        tmp_data[:, -1] = np.array([user_no for i in range(extracted_data.shape[0])])
        with open(join(OUTPUT_FOLDER, dataset_name, "user_{}_data_continuous.pickle".format(user_no)),
                  "wb") as output_file:
            pickle.dump(tmp_data, output_file)


def extract_unilever_drinking():
    dataset_name = "unilever_drinking"
    data = np.loadtxt(UNILEVER_DRINKING, delimiter=",")
    gestures = data[:, :-1]
    labels = data[:, -1]
    all_data = list()
    for i, g in enumerate(gestures):
        tmp_data = np.empty((len(g), 4))
        tmp_data[:, 0] = np.arange(len(g))
        tmp_data[:, 1] = g.T * 1000
        tmp_data[:, 2] = np.array([labels[i] for _ in range(len(g))])
        tmp_data[:, 3] = np.array([0 for _ in range(len(g))])
        all_data.append(tmp_data)
    with open(join(OUTPUT_FOLDER, dataset_name, "all_data_isolated.pickle"), "wb") as output_file:
        pickle.dump(all_data, output_file)


def extract_isolated_skoda_gestures(encoding=False, use_spatial=True, use_temporal=False, use_rotation=False):
    files = [file for file in glob.glob(SKODA_FOLDER + "/subject*.txt") if
             os.stat(file).st_size != 0]
    eng = matlab.engine.start_matlab()
    dataset_name = "skoda"
    all_data = list()
    for file in files:
        user_no = SKODA_USER_DICT[file.split("/")[-1].split("_")[0][-1]]
        data = np.loadtxt(file, dtype=float)
        if encoding:
            if use_spatial and use_temporal:
                data = compute_displacement(data, 27, use_rotation=use_rotation)
            elif use_spatial:
                data = compute_spatial_displacement(data, 27, use_rotation=use_rotation)
            else:
                data = compute_temporal_displacement(data, 27, use_rotation=use_rotation)
        labels = data[:, -1]
        [i1, i2, i3] = eng.dtcFindInstancesFromLabelStream(matlab.double(list(labels)), nargout=3)
        for i, c in enumerate(np.unique(labels)):
            m_range = i3[i]['range']
            num_inst = len(m_range)
            for k in range(num_inst):
                extracted_data = data[int(i3[i]['range'][k][0] - 1):int(i3[i]['range'][k][1]), 1]
                if extracted_data.size != 0:
                    extracted_data_time = data[int(i3[i]['range'][k][0] - 1):int(i3[i]['range'][k][1]), 0]
                    tmp_data = np.empty((extracted_data.shape[0], 4))
                    tmp_data[:, 0] = extracted_data_time
                    tmp_data[:, 1] = extracted_data
                    tmp_data[:, -2] = np.array([c for i in range(extracted_data.shape[0])])
                    tmp_data[:, -1] = np.array([user_no for i in range(extracted_data.shape[0])])
                    all_data.append(tmp_data)
    eng.quit()
    with open(join(OUTPUT_FOLDER, dataset_name, "all_data_isolated.pickle"), "wb") as output_file:
        pickle.dump(all_data, output_file)


def extract_old_skoda():
    data = np.loadtxt(SKODA_OLD_FILE)
    eng = matlab.engine.start_matlab()
    labels = data[:, -1]
    all_data = list()
    [i1, i2, i3] = eng.dtcFindInstancesFromLabelStream(matlab.double(list(labels)), nargout=3)
    dataset_name = "skoda"
    for i, c in enumerate(np.unique(labels)):
        m_range = i3[i]['range']
        num_inst = len(m_range)
        for k in range(num_inst):
            extracted_data = data[int(i3[i]['range'][k][0] - 1):int(i3[i]['range'][k][1]), 0]
            if extracted_data.size != 0:
                extracted_data_time = data[int(i3[i]['range'][k][0] - 1):int(i3[i]['range'][k][1]), 1]
                tmp_data = np.empty((extracted_data.shape[0], 4))
                tmp_data[:, 0] = extracted_data_time
                tmp_data[:, 1] = extracted_data.astype(np.int)
                tmp_data[:, -2] = np.array([c for i in range(extracted_data.shape[0])], dtype=int)
                tmp_data[:, -1] = np.array([0 for i in range(extracted_data.shape[0])], dtype=int)
                all_data.append(tmp_data)
    eng.quit()
    with open(join(OUTPUT_FOLDER, dataset_name, "all_old_data_isolated.pickle"), "wb") as output_file:
        pickle.dump(all_data, output_file)


def compute_displacement(data, alphabet, use_rotation=False):
    codebook = cc.create_3d_codebook(alphabet)
    v_hand_prev = None
    v_displ_list = []
    time = []
    labels = []
    for t in range(len(data)):
        if t % SKODA_RESCALING_FACTOR == 0:
            vlimbs = []
            for key in SKODA_DATA_IDX.keys():
                idx = SKODA_DATA_IDX[key]
                qi = Quaternion(data[t, idx[0]], data[t, idx[1]], data[t, idx[2]], data[t, idx[3]])
                vlimbs.append(qi.rotate(v))
            v_hand = np.sum(vlimbs, axis=0)
            if v_hand_prev is None:
                v_hand_prev = v_hand
            else:
                v_displ = np.subtract(v_hand, v_hand_prev)
                v_hand_prev = v_hand
                n = np.linalg.norm(v_displ)
                if n != 0:
                    v_displ_norm = v_displ / n
                else:
                    v_displ_norm = v_displ
                v_displ_list.append(cc.code_vector(codebook, tuple(v_displ_norm)))
                time.append(data[t, 0])
                labels.append(data[t, 53])
    tmp_data = [x for x in zip(time, v_displ_list, labels)]
    return tmp_data


def compute_spatial_displacement(data, alphabet, use_rotation=False):
    codebook = cc.create_3d_codebook(num=alphabet)
    v = np.array([0., 1., 0.])
    v_hand_prev = None
    prev_label = None
    v_displ_list = []
    time = []
    labels = []
    for t in range(len(data)):
        if t % SKODA_RESCALING_FACTOR == 0:
            vlimbs = []
            timestamp = data[t, 0]
            label = data[t, 113]
            for key in SKODA_DATA_IDX.keys():
                idx = SKODA_DATA_IDX[key]
                qi = Quaternion(data[t, idx[0]], data[t, idx[1]], data[t, idx[2]], data[t, idx[3]])
                vlimbs.append(qi.rotate(v))
            v_hand = np.sum(vlimbs, axis=0)
            if v_hand_prev is None:
                v_hand_prev = v_hand
                prev_label = label
            else:
                v_displ = np.subtract(v_hand, v_hand_prev)
                n = np.linalg.norm(v_displ)
                if n > SKODA_MIN_DISPLACEMENT or label != prev_label:
                    v_displ_norm = v_displ / n
                    v_hand_prev = v_hand
                    v_displ_list.append(cc.code_vector(codebook, tuple(v_displ_norm)))
                    time.append(timestamp)
                    labels.append(label)
                    prev_label = label
    tmp_data = np.array([x for x in zip(time, v_displ_list, labels)])
    return tmp_data


def compute_temporal_displacement(data, alphabet, use_rotation=False):
    codebook = cc.create_3d_codebook(num=alphabet)
    v = np.array([0., 1., 0.])
    v_hand_prev = None
    prev_time = None
    prev_label = None
    v_displ_list = []
    time = []
    labels = []
    for t in range(len(data)):
        vlimbs = []
        timestamp = float(data[t, 0])
        label = data[t, 53]
        for key in SKODA_DATA_IDX.keys():
            idx = SKODA_DATA_IDX[key]
            qi = Quaternion(data[t, idx[0]], data[t, idx[1]], data[t, idx[2]], data[t, idx[3]])
            vlimbs.append(qi.rotate(v))
        v_hand = np.sum(vlimbs, axis=0)
        if v_hand_prev is None:
            v_hand_prev = v_hand
            prev_time = timestamp
        else:
            if (timestamp - prev_time) >= SKODA_TIME_INTERVAL or prev_label != label:
                v_displ = np.subtract(v_hand, v_hand_prev)
                v_hand_prev = v_hand
                n = np.linalg.norm(v_displ)
                if n != 0:
                    v_displ_norm = v_displ / n
                else:
                    v_displ_norm = v_displ
                v_displ_list.append(cc.code_vector(codebook, tuple(v_displ_norm)))
                time.append(data[t, 0])
                labels.append(data[t, 53])
    tmp_data = [x for x in zip(time, v_displ_list, labels)]
    return tmp_data


def create_synthetic_dataset(num_gestures=20, gesture_length=150):
    gestures = list()
    dataset_name = "synthetic"
    gestures += generate_sinusoid(num_gestures, gesture_length)
    gestures += generate_normal_dist(num_gestures, gesture_length)
    gestures += generate_step_function(num_gestures, gesture_length)
    gestures += generate_squared_impulse(num_gestures, gesture_length)
    with open(join(OUTPUT_FOLDER, dataset_name, "all_data_isolated.pickle"), "wb") as output_file:
        pickle.dump(gestures, output_file)


def generate_sinusoid(num_gestures, gesture_length):
    sin_length = int(gesture_length / 3)
    data = [np.zeros(gesture_length) for _ in range(num_gestures)]
    f = 1
    w = 2 * np.pi * f
    for i in range(num_gestures):
        start_idx = int(np.random.uniform(0, int(gesture_length / 2)))
        end_idx = start_idx + sin_length
        t = np.linspace(0, 1, sin_length)
        tmp_sin = np.sin(w * t) * np.random.uniform(48, 64)
        data[i][start_idx:end_idx] = tmp_sin
    time = np.arange(gesture_length)
    label = np.array([1001 for _ in range(gesture_length)])
    user_no = np.array([0 for _ in range(gesture_length)])
    generated_data = [np.stack((time, d, label, user_no), axis=-1) for d in data]
    return generated_data


def generate_normal_dist(num_gestures, gesture_length):
    t_mean = np.random.randint(int(gesture_length / 5), int(gesture_length / 5 * 3), size=num_gestures)
    t_std = np.random.randint(int(gesture_length / 7 / 10), int(gesture_length / 7 / 10 * 3), size=num_gestures)
    data = np.array([norm.pdf(np.arange(gesture_length), x, y) * 900 for x, y in zip(t_mean, t_std)])
    time = np.arange(gesture_length)
    label = np.array([1002 for _ in range(gesture_length)])
    user_no = np.array([0 for _ in range(gesture_length)])
    generated_data = [np.stack((time, d, label, user_no), axis=-1) for d in data]
    return generated_data


def generate_step_function(num_gestures, gesture_length):
    t_mean = np.random.randint(int(gesture_length / 5), int(gesture_length / 5 * 3), size=num_gestures)
    t_std = np.random.randint(int(gesture_length / 5 / 10), int(gesture_length / 5 / 10 * 3), size=num_gestures)
    step_heigth = np.random.randint(20, 100, size=num_gestures)
    data = np.array([uniform.cdf(np.arange(gesture_length), x, y) * st for x, y, st in zip(t_mean, t_std, step_heigth)])
    time = np.arange(gesture_length)
    label = np.array([1003 for _ in range(gesture_length)])
    user_no = np.array([0 for _ in range(gesture_length)])
    generated_data = [np.stack((time, d, label, user_no), axis=-1) for d in data]
    return generated_data


def generate_squared_impulse(num_gestures, gesture_length):
    t_mean = np.random.randint(int(gesture_length / 5), int(gesture_length / 5 * 3), size=num_gestures)
    t_std = np.random.randint(int(gesture_length / 4 / 2), int(gesture_length / 4 / 2 * 3), size=num_gestures)
    data = np.array([uniform.pdf(np.arange(gesture_length), x, y) * 900 for x, y in zip(t_mean, t_std)])
    time = np.arange(gesture_length)
    label = np.array([1004 for _ in range(gesture_length)])
    user_no = np.array([0 for _ in range(gesture_length)])
    generated_data = [np.stack((time, d, label, user_no), axis=-1) for d in data]
    return generated_data

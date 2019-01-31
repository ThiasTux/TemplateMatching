import glob
import os
from os.path import expanduser
from os.path import join
import matlab.engine
import numpy as np
import pickle

OPPORTUNITY_FOLDER = join(expanduser("~"), "Documents/Datasets/OpportunityUCIDataset/dataset")
SKODA_FOLDER = join(expanduser("~"), "Documents/Datasets/SkodaDataset/processed_data/")
SKODA_USER_DICT = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7}
SKODA_COLS = [np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 14, 15, 16]) + (i * 16) for i in range(7)]
SKODA_COLS = [item for sublist in SKODA_COLS for item in sublist]
OUTPUT_FOLDER = "outputs/datasets/"


def extract_isolated_opportunity():
    files = [file for file in glob.glob(OPPORTUNITY_FOLDER + "/*-Drill.dat") if
             os.stat(file).st_size != 0]
    eng = matlab.engine.start_matlab()
    dataset_name = "opportunity"
    all_data = list()
    for file in files:
        user_no = file.split("/")[-1].replace("-Drill.dat", "").replace("S", "0")
        data = np.loadtxt(file, dtype=float)
        labels = data[:, -1]
        [i1, i2, i3] = eng.dtcFindInstancesFromLabelStream(matlab.double(list(labels)), nargout=3)
        for i, c in enumerate(np.unique(labels)):
            m_range = i3[i]['range']
            num_inst = len(m_range)
            for k in range(num_inst):
                extracted_data = data[int(i3[i]['range'][k][0] - 1):int(i3[i]['range'][k][1] - 1), 37:102]
                if extracted_data.size != 0:
                    extracted_data_time = data[int(i3[i]['range'][k][0] - 1):int(i3[i]['range'][k][1] - 1), 0]
                    tmp_data = np.empty((extracted_data.shape[0], extracted_data.shape[1] + 3))
                    tmp_data[:, 0] = extracted_data_time
                    tmp_data[:, 1:-2] = extracted_data
                    tmp_data[:, -2] = np.array([c for i in range(extracted_data.shape[0])])
                    tmp_data[:, -1] = np.array([user_no for i in range(extracted_data.shape[0])])
                    all_data.append(tmp_data)
    eng.quit()
    with open(join(OUTPUT_FOLDER, dataset_name, "all_data_isolated.pickle"), "wb") as output_file:
        pickle.dump(all_data, output_file)


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
                enc_data = compute_displacement(use_rotation=use_rotation)
            elif use_spatial:
                enc_data = compute_spatial_displacement(use_rotation=use_rotation)
            else:
                enc_data = compute_temporal_displacement(use_rotation=use_rotation)
        else:
            labels = data[:, -1]
            [i1, i2, i3] = eng.dtcFindInstancesFromLabelStream(matlab.double(list(labels)), nargout=3)
            for i, c in enumerate(np.unique(labels)):
                m_range = i3[i]['range']
                num_inst = len(m_range)
                for k in range(num_inst):
                    extracted_data = data[int(i3[i]['range'][k][0] - 1):int(i3[i]['range'][k][1] - 1), SKODA_COLS]
                    if extracted_data.size != 0:
                        extracted_data_time = data[int(i3[i]['range'][k][0] - 1):int(i3[i]['range'][k][1] - 1), 0]
                        tmp_data = np.empty((extracted_data.shape[0], extracted_data.shape[1] + 3))
                        tmp_data[:, 0] = extracted_data_time
                        tmp_data[:, 1:-2] = extracted_data
                        tmp_data[:, -2] = np.array([c for i in range(extracted_data.shape[0])])
                        tmp_data[:, -1] = np.array([user_no for i in range(extracted_data.shape[0])])
                        all_data.append(tmp_data)
    eng.quit()
    with open(join(OUTPUT_FOLDER, dataset_name, "all_data_isolated.pickle"), "wb") as output_file:
        pickle.dump(all_data, output_file)


def compute_displacement(use_rotation=False):
    pass


def compute_spatial_displacement(use_rotation=False):
    pass


def compute_temporal_displacement(use_rotation=False):
    pass

import pickle
import random

import numpy as np

from utils import distance_measures as dtm


def load_dataset(dataset_choice=100, classes=None, num_gestures=None, user=None):
    # Skoda dataset
    if dataset_choice == 100:
        data = pickle.load(open("outputs/datasets/skoda/all_data_isolated.pickle", "rb"))
    elif dataset_choice == 101:
        data = pickle.load(open("outputs/datasets/skoda/all_old_data_isolated.pickle", "rb"))
    # Opportunity dataset
    elif dataset_choice == 200:
        data = pickle.load(open("outputs/datasets/opportunity/all_data_isolated.pickle", "rb"))
    elif dataset_choice == 201:
        data = pickle.load(open("outputs/datasets/opportunity/all_quant_accy_data_isolated.pickle", "rb"))
    elif dataset_choice == 211:
        data = pickle.load(open("outputs/datasets/opportunity/all_old_data_isolated.pickle", "rb"))
    # HCI guided
    elif dataset_choice == 300:
        data = pickle.load(open("outputs/datasets/hci/all_data_isolated.pickle", "rb"))
    # Synthetic datasets
    elif dataset_choice == 700:
        data = pickle.load(open("outputs/datasets/synthetic/all_data_isolated.pickle", "rb"))
    elif dataset_choice == 701:
        data = pickle.load(open("outputs/datasets/synthetic2/all_data_isolated.pickle", "rb"))
    elif dataset_choice == 702:
        data = pickle.load(open("outputs/datasets/synthetic3/all_data_isolated.pickle", "rb"))
    elif dataset_choice == 704:
        data = pickle.load(open("outputs/datasets/synthetic4/all_data_isolated.pickle", "rb"))
    elif dataset_choice == 800:
        data = pickle.load(open("outputs/datasets/unilever_drinking/all_data_isolated.pickle", "rb"))
    if user is None:
        selected_data = [[d for d in data if d[0, -2] == c] for c in classes]
    else:
        selected_data = [[d for d in data if d[0, -2] == c and d[0, -1] == user] for c in classes]
    if num_gestures is not None:
        try:
            selected_data = [random.sample(sel_data, num_gestures) for sel_data in selected_data]
        except ValueError:
            pass
    return [instance for class_data in selected_data for instance in class_data]


def load_training_dataset(dataset_choice=704, classes=None, num_gestures=None, user=None, extract_null=False,
                          null_class_percentage=0.5, template_choice_method=1, seed=2):
    # Skoda dataset
    if dataset_choice == 100:
        data = pickle.load(open("outputs/datasets/skoda/all_data_isolated.pickle", "rb"))
    elif dataset_choice == 101:
        data = pickle.load(open("outputs/datasets/skoda/all_old_data_isolated.pickle", "rb"))
    # Opportunity dataset
    elif dataset_choice == 200:
        data = pickle.load(open("outputs/datasets/opportunity/all_data_isolated.pickle", "rb"))
    elif dataset_choice == 201:
        data = pickle.load(open("outputs/datasets/opportunity/all_quant_accy_data_isolated.pickle", "rb"))
    # HCI guided
    elif dataset_choice == 300:
        data = pickle.load(open("outputs/datasets/hci/all_data_isolated.pickle", "rb"))
    # Synthetic dataset
    elif dataset_choice == 700:
        data = pickle.load(open("outputs/datasets/synthetic/all_data_isolated.pickle", "rb"))
    elif dataset_choice == 701:
        data = pickle.load(open("outputs/datasets/synthetic2/all_data_isolated.pickle", "rb"))
    elif dataset_choice == 702:
        data = pickle.load(open("outputs/datasets/synthetic3/all_data_isolated.pickle", "rb"))
    elif dataset_choice == 704:
        data = pickle.load(open("outputs/datasets/synthetic4/all_data_isolated.pickle", "rb"))
    elif dataset_choice == 800:
        data = pickle.load(open("outputs/datasets/unilever_drinking/all_data_isolated.pickle", "rb"))
    if user is None:
        selected_data = [[d for d in data if d[0, -2] == c] for c in classes]
    else:
        selected_data = [[d for d in data if d[0, -2] == c and d[0, -1] == user] for c in classes]
    if num_gestures is not None:
        try:
            selected_data = [random.sample(sel_data, num_gestures) for sel_data in selected_data]
        except ValueError:
            pass
    labels = [instance[0, -2] for class_data in selected_data for instance in class_data]

    if template_choice_method != 0:
        chosen_templates = [np.array([]) for _ in classes]
        if template_choice_method == 1:
            for k, c in enumerate(classes):
                templates = selected_data[k]
                matching_scores = np.zeros((len(templates), len(templates)), dtype=int)
                for i in range(len(templates)):
                    for j in range(i + 1, len(templates)):
                        d, c = dtm.LCS(templates[i][:, 1], templates[j][:, 1])
                        matching_scores[i][j] = d
                        matching_scores[j][i] = d
                matching_scores_sums = np.sum(matching_scores, axis=0)
                matching_scores_perc = np.array(
                    [matching_scores_sums[i] / len(templates[i]) for i in range(len(templates))])
                ordered_indexes = np.argsort(matching_scores_perc)
                chosen_templates[k] = np.array(templates[ordered_indexes[-1]])
        elif template_choice_method == 2:
            for k, c in enumerate(classes):
                templates = [d for d in selected_data if d[0, -2] == c]
                chosen_templates[k] = templates[np.random.uniform(0, len(templates))]
        templates = [instance for class_data in selected_data for instance in class_data]
        if extract_null:
            tmp_null_selected_data = [d for d in data if d[0, -2] == 0]
            null_class_data = [item for d in tmp_null_selected_data for item in d[:, 1]]
            num_null_instances = int((len(templates) * null_class_percentage) / (1 - null_class_percentage))
            null_selected_data = list()
            avg_length = int(np.average([len(d) for d in selected_data]))
            for i in range(num_null_instances):
                tmp_null_data = np.zeros((avg_length, 4))
                tmp_null_data[:, 0] = np.arange(avg_length)
                np.random.seed(2)
                start_idx = np.random.randint(0, len(null_class_data) - avg_length)
                end_idx = start_idx + avg_length
                tmp_null_data[:, 1] = null_class_data[start_idx:end_idx]
                null_selected_data.append(tmp_null_data)
            null_labels = [0 for _ in null_selected_data]
            labels += null_labels
            templates += null_selected_data
        return chosen_templates, templates, labels
    else:
        templates = [instance for class_data in selected_data for instance in class_data]
        if extract_null:
            tmp_null_selected_data = [d for d in data if d[0, -2] == 0]
            null_class_data = [item for d in tmp_null_selected_data for item in d[:, 1]]
            num_null_instances = int((len(templates) * null_class_percentage) / (1 - null_class_percentage))
            null_selected_data = list()
            avg_length = int(np.average([len(d) for d in selected_data]))
            for i in range(num_null_instances):
                tmp_null_data = np.zeros((avg_length, 4))
                tmp_null_data[:, 0] = np.arange(avg_length)
                start_idx = np.random.randint(0, len(null_class_data) - avg_length)
                end_idx = start_idx + avg_length
                tmp_null_data[:, 1] = null_class_data[start_idx:end_idx]
                null_selected_data.append(tmp_null_data)
            null_labels = [0 for _ in null_selected_data]
            labels += null_labels
            templates += null_selected_data
        return templates, labels


def load_evolved_templates(es_results_file, classes, use_evolved_thresholds=False):
    chosen_templates = [None for _ in classes]
    thresholds = list()
    for i, c in enumerate(classes):
        file_path = es_results_file + "_00_{}_templates.txt".format(c)
        with open(file_path, "r") as templates_file:
            last_line = templates_file.readlines()[-1]
            if use_evolved_thresholds:
                template = np.array([int(v) for v in last_line.split(" ")[:-1]])
                thresholds.append(int(last_line.split(" ")[-1]))
            else:
                template = np.array([int(v) for v in last_line.split(" ")])
            chosen_templates[i] = np.stack((np.arange(len(template)), template), axis=-1)
    if use_evolved_thresholds:
        return chosen_templates, thresholds
    else:
        return chosen_templates


def load_continuous_dataset(dataset_choice='skoda', user=1, template_choice_method=1, seed=2):
    # Skoda dataset
    if dataset_choice == 'skoda':
        data = pickle.load(open("outputs/datasets/skoda/all_data_isolated.pickle", "rb"))
    elif dataset_choice == 'skoda_old':
        data = pickle.load(open("outputs/datasets/skoda/all_old_data_isolated.pickle", "rb"))
    # Opportunity dataset
    elif dataset_choice == 'opp':
        data = pickle.load(
            open("outputs/datasets/opportunity/user_{:02d}_accx_data_continuous.pickle".format(user), "rb"))
    elif dataset_choice == 'opp_quant':
        data = pickle.load(
            open("outputs/datasets/opportunity/user_{:02d}_quant_accy_data_continuous.pickle".format(user), "rb"))
    # HCI guided
    elif dataset_choice == 'hci':
        data = pickle.load(
            open("outputs/datasets/hci/user_{:02d}_data_continuous.pickle".format(user), "rb"))
    # Synthetic dataset
    elif dataset_choice == 'synt_1':
        data = pickle.load(open("outputs/datasets/synthetic/all_data_isolated.pickle", "rb"))
    elif dataset_choice == 'synt_2':
        data = pickle.load(open("outputs/datasets/synthetic2/all_data_isolated.pickle", "rb"))
    elif dataset_choice == 'unil':
        data = pickle.load(open("outputs/datasets/unilever_drinking/all_data_isolated.pickle", "rb"))
    return data


def enc_data_loader(input_path):
    data = np.loadtxt(input_path, )
    return data

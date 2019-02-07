import pickle
import random

import numpy as np

from utils import distance_measures as dtm


def load_dataset(dataset_choice=100, classes=None, num_gestures=None, user=None):
    # Skoda dataset
    if dataset_choice == 100:
        data = pickle.load(open("outputs/datasets/skoda/all_data_isolated.pickle", "rb"))
    # Opportunity dataset
    elif dataset_choice == 200:
        data = pickle.load(open("outputs/datasets/opportunity/all_data_isolated.pickle", "rb"))
    elif dataset_choice == 201:
        data = pickle.load(open("outputs/datasets/opportunity/all_quant_data_isolated.pickle", "rb"))
    # Synthetic dataset
    elif dataset_choice == 700:
        data = pickle.load(open("outputs/datasets/synthetic/all_data_isolated.pickle", "rb"))
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
    # HCI guided
    # elif dataset_choice == 300:
    #
    #     filename = join(homefolder, "Documents/Datasets/FreeHandGestures/usb_hci_guided.csv")
    #     data = load_hci_dataset(filename, sensor)
    # # HCI freehand
    # elif dataset_choice == 400:
    #     filename = join(homefolder, "Documents/Datasets/FreeHandGestures/usb_hci_freehand.csv")
    #     data = load_hci_dataset(filename, sensor)
    # notMNIST
    # elif dataset_choice == 500:
    #     np.random.seed(2)
    #     subset_num = 10
    #     letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    #     flat_instances = list()
    #     templates = list()
    #     labels = list()
    #     chosen_templates = list()
    #     for c in classes:
    #         data = pickle.load(open(join(notMNIST_small, "{}.pickle".format(letters[c])), 'rb'))
    #         subidxs = np.random.choice(len(data), subset_num)
    #         subset = np.array(data[subidxs] * 1000, dtype=int)
    #         labels += [c for x in range(subset_num)]
    #         tmp_templates = [pd.Series(x.flatten()) for x in subset]
    #         flat_instances += tmp_templates
    #         templates.append(tmp_templates)
    #         chosen_templates.append(subset[0].flatten())
    #     return flat_instances, labels, chosen_templates, templates, list()
    # if isolated:
    #     return wlp.extract_instances(data, classes, get_templates=True, extract_null=extract_null,
    #                                  null_class_percentage=null_class_percentage,
    #                                  template_choice_method=template_choice_method, seed=seed)
    # else:
    #     flat_instances, labels, chosen_templates, templates, instances = wlp.extract_instances(data, classes,
    #                                                                                            get_templates=True)
    #     return data['edrha'].values, data['label'].values, chosen_templates, templates, data['time'].values


def load_training_dataset(dataset_choice=700, classes=None, num_gestures=None, user=None, extract_null=False,
                          null_class_percentage=0.5, template_choice_method=1, seed=2):
    # Skoda dataset
    if dataset_choice == 100:
        data = pickle.load(open("outputs/datasets/skoda/all_data_isolated.pickle", "rb"))
    # Opportunity dataset
    elif dataset_choice == 200:
        data = pickle.load(open("outputs/datasets/opportunity/all_data_isolated.pickle", "rb"))
    elif dataset_choice == 201:
        data = pickle.load(open("outputs/datasets/opportunity/all_quant_data_isolated.pickle", "rb"))
    # Synthetic dataset
    elif dataset_choice == 700:
        data = pickle.load(open("outputs/datasets/synthetic/all_data_isolated.pickle", "rb"))
    if user is None:
        selected_data = [[d for d in data if d[0, -2] == c] for c in classes]
    else:
        selected_data = [[d for d in data if d[0, -2] == c and d[0, -1] == user] for c in classes]
    if num_gestures is not None:
        try:
            selected_data = [random.sample(sel_data, num_gestures) for sel_data in selected_data]
        except ValueError:
            pass
    chosen_templates = [np.array([]) for _ in classes]
    if template_choice_method == 1:
        for k, c in enumerate(classes):
            templates = selected_data[k]
            matching_scores = np.zeros((len(templates), len(templates)), dtype=int)
            for i in range(len(templates)):
                for j in range(i + 1, len(templates)):
                    d, c = dtm.LCS(templates[i][:, 0], templates[j][:, 0])
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
    labels = [instance[0, -2] for class_data in selected_data for instance in class_data]
    return chosen_templates, templates, labels


def enc_data_loader(input_path):
    data = np.loadtxt(input_path, )
    return data

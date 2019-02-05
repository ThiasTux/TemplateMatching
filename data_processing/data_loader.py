import pickle
import random

import numpy as np


def load_dataset(dataset_choice=100, classes=None, num_gestures=None, user=None, isolated=True, sensor=32,
                 extract_null=False,
                 null_class_percentage=0.5, template_choice_method=1, seed=2):
    # Skoda dataset
    if dataset_choice == 100:
        data = pickle.load(open("outputs/datasets/skoda/all_data_isolated.pickle", "rb"))
    # Opportunity dataset
    elif dataset_choice == 200:
        data = pickle.load(open("outputs/datasets/opportunity/all_data_isolated.pickle", "rb"))
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


def enc_data_loader(input_path):
    data = np.loadtxt(input_path, )
    return data

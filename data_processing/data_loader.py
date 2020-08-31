import numpy as np

from data_processing.hci import HCIGuided
from data_processing.hcitable import HCITable
from data_processing.opportunity import OpportunityDataset, OpportunityDatasetEncoded
from data_processing.shl import SHLPreview
from data_processing.skoda import Skoda
from data_processing.skoda_mini import SkodaMini
from data_processing.synthetic import Synthetic1, Synthetic2, Synthetic3, Synthetic4
from data_processing.ucr_dataset import UWaveGestureLibraryX
from data_processing.utd_mhad import UTDMhad
from utils import distance_measures as dtm


def load_training_dataset(dataset_choice='opportunity', template_choice_method='random', classes=None, seed=42,
                          train_test_split=0.5,
                          extract_null=False, null_class_percentage=0.5, num_gestures=None, user=None):
    # Load the dataset
    if dataset_choice == 'skoda':
        dataset = Skoda()
    elif dataset_choice == 'skoda_mini':
        dataset = SkodaMini()
    elif dataset_choice == 'opportunity':
        dataset = OpportunityDataset()
    elif dataset_choice == 'opportunity_encoded':
        dataset = OpportunityDatasetEncoded()
    elif dataset_choice == 'utd_mhad':
        dataset = UTDMhad()
    elif dataset_choice == 'hci_guided':
        dataset = HCIGuided()
    elif dataset_choice == 'hci_table':
        dataset = HCITable()
    elif dataset_choice == 'synthetic1':
        dataset = Synthetic1()
    elif dataset_choice == 'synthetic2':
        dataset = Synthetic2()
    elif dataset_choice == 'synthetic3':
        dataset = Synthetic3()
    elif dataset_choice == 'synthetic4':
        dataset = Synthetic4()
    elif dataset_choice == 'shl_preview':
        dataset = SHLPreview()
    elif dataset_choice == 'uwave_x':
        dataset = UWaveGestureLibraryX()

    streams, labels = dataset.load_isolated_dataset()

    # Fix null class
    if extract_null:
        pass
    else:
        streams = [streams[t] for t in np.where(labels > 0)[0]]
        labels = labels[np.where(labels > 0)[0]]
    templates = list()

    # Select template for matching
    if classes is None:
        classes = np.unique(labels)
        classes = np.delete(classes, np.argwhere(classes == 0))
    # Random choice
    np.random.seed(seed)
    if template_choice_method is None:
        return streams, labels
    elif template_choice_method == 'random':
        for c in classes:
            templates.append(streams[np.random.choice(np.where(labels == c)[0])])
    elif template_choice_method == 'mrt_lcs':
        for k, c in enumerate(classes):
            tmp_templates = [streams[t] for t in np.where(labels == c)[0]]
            matching_scores = np.zeros((len(tmp_templates), len(tmp_templates)), dtype=int)
            for i in range(len(tmp_templates)):
                for j in range(i + 1, len(tmp_templates)):
                    d, c = dtm.LCS(tmp_templates[i], tmp_templates[j])
                    matching_scores[i][j] = d
                    matching_scores[j][i] = d
            matching_scores_sums = np.sum(matching_scores, axis=0)
            matching_scores_perc = np.array(
                [matching_scores_sums[i] / len(tmp_templates[i]) for i in range(len(tmp_templates))])
            ordered_indexes = np.argsort(matching_scores_perc)
            templates.append(np.array(tmp_templates[ordered_indexes[-1]]))
    elif template_choice_method == 'mrt_dtw':
        pass

    return templates, streams, labels


def load_continuous_dataset(dataset_choice='opportunity', template_choice_method='random', classes=None, seed=42,
                            extract_null=False, null_class_percentage=0.5, num_gestures=None, user=None):
    # Load the dataset
    if dataset_choice == 'skoda':
        dataset = Skoda()
    elif dataset_choice == 'skoda_mini':
        dataset = SkodaMini()
    elif dataset_choice == 'opportunity':
        dataset = OpportunityDataset()
    elif dataset_choice == 'utd_mhad':
        dataset = UTDMhad()
    elif dataset_choice == 'hci_guided':
        dataset = HCIGuided()
    elif dataset_choice == 'hci_table':
        dataset = HCITable()
    elif dataset_choice == 'synthetic1':
        dataset = Synthetic1()
    elif dataset_choice == 'synthetic2':
        dataset = Synthetic2()
    elif dataset_choice == 'synthetic3':
        dataset = Synthetic3()
    elif dataset_choice == 'synthetic4':
        dataset = Synthetic4()
    elif dataset_choice == 'shl_preview':
        dataset = SHLPreview()

    streams, labels, timestamps = dataset.load_continuous_dataset()

    return streams, labels, timestamps


def load_evolved_templates(es_results_file, classes, use_evolved_thresholds=False):
    chosen_templates = [None for _ in classes]
    thresholds = list()
    for i, c in enumerate(classes):
        file_path = es_results_file + "_{}_templates.txt".format(c)
        with open(file_path, "r") as templates_file:
            last_line = templates_file.readlines()[-1]
            if use_evolved_thresholds:
                template = np.array([int(v) for v in last_line.split(" ")[:-1]])
                thresholds.append(int(last_line.split(" ")[-1]))
            else:
                template = np.array([int(v) for v in last_line.split(" ")])
            chosen_templates[i] = template
    if use_evolved_thresholds:
        return chosen_templates, thresholds
    else:
        return chosen_templates


def load_params(ga_results_file):
    params = []
    with open(ga_results_file, 'r') as inputfile:
        for line in inputfile.readlines():
            tmp_line = line.replace("[", "").replace("]", "")
            line_values = tmp_line.split(",")
            p = [int(i) for i in line_values[:3]]
            params.append(p)
    return params


def load_thresholds(ga_results_file):
    thresholds = []
    with open(ga_results_file, 'r') as inputfile:
        for line in inputfile.readlines():
            tmp_line = line.replace("[", "").replace("]", "")
            line_values = tmp_line.split(",")
            thresholds.append(int(line_values[3]))
    return thresholds

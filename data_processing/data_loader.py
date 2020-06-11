from data_processing.hcitable import HCITable
from data_processing.opportunity import OpportunityDataset
from data_processing.skoda import Skoda
from data_processing.skoda_mini import SkodaMini
from data_processing.synthetic import Synthetic1, Synthetic2, Synthetic3, Synthetic4
from data_processing.utd_mhad import UTDMhad
from data_processing.hci import HCIGuided

import numpy as np


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
        pass
    elif template_choice_method == 'mrt_dtw':
        pass

    return templates, streams, labels

#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np

from data_processing import data_loader as dl
from performance_evaluation import performance_evaluation as pfe
from template_matching.utils import find_peaks
from template_matching.wlcss_cuda_class import WLCSSCudaParamsTraining, WLCSSCudaContinuous
from utils.plots import plot_creator as plt_creator

if __name__ == '__main__':
    dataset_choice = 'skoda'
    outputs_path = "/home/mathias/Documents/Academic/PhD/Research/WLCSSTraining/training/cuda"

    isolated_case = True  # True for isolate, False for continuous
    save_img = False
    use_encoding = False

    use_null = False
    user = None
    use_evolved_templates = False
    use_evolved_thresholds = False

    write_to_file = True
    if dataset_choice == 'skoda':
        use_encoding = '3d'
        # classes = [3001, 3003, 3013, 3018]
        classes = [3001, 3002, 3003, 3005, 3013, 3014, 3018, 3019]
        output_folder = "{}/skoda/params".format(outputs_path)
        params = [57, 2, 8]
        thresholds = [370, 353, 220, 233, 307, 463, 228, 135]
        null_class_percentage = 0.6
        wsize = 500
        temporal_merging_window = 50
        tolerance_window = 10
        es_results_file = "{}/skoda/templates/zeus_templates_2020-07-29_17-54-30".format(outputs_path)
    elif dataset_choice == 'skoda_mini':
        use_encoding = False
        classes = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57]
        output_folder = "{}/skoda_mini/params".format(outputs_path)
        null_class_percentage = 0.6
        params = [25, 6, 1]
        thresholds = [22, 169, 29, 283, 284, 311, 225, 138]
        es_results_file = "{}/skoda_mini/templates/templates_2019-03-27_18-24-04".format(outputs_path)
        wsize = 1000
        temporal_merging_window = 5
        tolerance_window = 5
    elif dataset_choice == 'opportunity':
        classes = [406516, 404516, 406505, 404505, 406519, 404519, 407521, 405506]
        # classes = [406516, 408512, 405506]
        # classes = [407521, 406520, 406505, 406519]
        output_folder = "{}/opportunity/params".format(outputs_path)
        user = 3
        params = [33, 9, 2]
        thresholds = [480, 2036, 921, 2038, 815, 1477, 1797, 0]
        wsize = 500
        es_results_file = "{}/opportunity/templates/templates_2019-04-17_16-33-36".format(outputs_path)
    elif dataset_choice == 'hci_guided':
        classes = [49, 50]
        output_folder = "{}/hci_guided/params".format(outputs_path)
        params = [31, 11, 4]
        thresholds = [2, 118]
        wsize = 5
        temporal_merging_window = 5
        null_class_percentage = 0.5
        es_results_file = "{}/hci_guided/templates/zeus_templates_2020-07-30_13-59-45".format(outputs_path)
    elif dataset_choice == 'hci_freehand':
        classes = [49, 50, 51, 52, 53]
        output_folder = "{}/hci_freehand/params".format(outputs_path)
    elif dataset_choice == 500:
        classes = [0, 7]
        output_folder = "{}/notmnist/params".format(outputs_path)
        null_class_percentage = 0
    elif dataset_choice == 'synthetic1':
        classes = [1001, 1002, 1003, 1004]
        output_folder = "{}/synthetic/params".format(outputs_path)
        null_class_percentage = 0
        params = [7, 5, 1]
        thresholds = [-2500, -2000, -4000, -2200]
        es_results_file = "{}/synthetic/templates/templates_2019-03-08_12-04-51".format(outputs_path)
    elif dataset_choice == 'synthetic2':
        classes = [1001, 1002]
        output_folder = "{}/synthetic2/params".format(outputs_path)
        null_class_percentage = 0
        params = [28, 2, 0]
        thresholds = [991, 567]
        es_results_file = "{}/synthetic2/templates/templates_2019-04-11_16-58-37".format(outputs_path)
    elif dataset_choice == 'synthetic4':
        classes = [1001, 1002, 1003, 1004]
        output_folder = "{}/synthetic4/params".format(outputs_path)
        null_class_percentage = 0
        params = [60, 4, 0]
        thresholds = [5534, 165, 3058, 4534]
        es_results_file = "{}/synthetic4/templates/zeus_templates_2020-07-28_23-30-25".format(outputs_path)
    elif dataset_choice == 'hci_table':
        encoding = '2d'
        classes = [i for i in range(1, 35)]
        output_folder = "{}/hci_table/params".format(outputs_path)
        null_class_percentage = 0.5
        params = [60, 4, 0]
        thresholds = [5534, 165, 3058, 4534]
        es_results_file = "{}/hci_table/templates/templates_2019-04-11_16-58-37".format(outputs_path)
    elif dataset_choice == 'shl_preview':
        classes = [1, 2, 4, 7, 8]
        output_folder = "{}/shl_preview/params".format(outputs_path)
        null_class_percentage = 0.5
        params = [60, 4, 0]
        thresholds = [5534, 165, 3058, 4534]
        es_results_file = "{}/shl_preview/templates/templates_2019-04-11_16-58-37".format(outputs_path)
    elif dataset_choice == 'uwave_x':
        output_folder = "{}/uwave_x/params".format(outputs_path)
        classes = [1]
        null_class_percentage = 0.5
        params = [60, 4, 0]
        thresholds = [5534, 165, 3058, 4534]
        es_results_file = "{}/uwave_x/templates/templates_2019-04-11_16-58-37".format(outputs_path)

    if isolated_case:
        templates, streams, streams_labels = dl.load_training_dataset(dataset_choice=dataset_choice, classes=classes,
                                                                      template_choice_method='mrt_lcs')
        # Group streams by labels
        streams_labels_sorted_idx = streams_labels.argsort()
        streams = [streams[i] for i in streams_labels_sorted_idx]
        streams_labels = streams_labels[streams_labels_sorted_idx]
    else:
        if not use_evolved_templates:
            templates, _, _ = dl.load_training_dataset(dataset_choice=dataset_choice, classes=classes,
                                                       template_choice_method='mrt_lcs')
        stream, labels, timestamps = dl.load_continuous_dataset(dataset_choice=dataset_choice, user=user)

    print("Data loaded!")

    if use_evolved_templates:
        if es_results_file is not None:
            if use_evolved_thresholds:
                templates, thresholds = dl.load_evolved_templates(es_results_file, classes,
                                                                  use_evolved_thresholds)
            else:
                templates = dl.load_evolved_templates(es_results_file, classes)
        print("Templates loaded!")

    if isolated_case:
        m_wlcss_cuda = WLCSSCuda(templates, streams, params, use_encoding)
        mss = m_wlcss_cuda.compute_wlcss()[0]
        m_wlcss_cuda.cuda_freemem()
        pfe.performance_evaluation_isolated(mss, streams_labels, thresholds, classes)
        plt_creator.plot_isolated_mss(mss, thresholds, dataset_choice, classes, streams_labels=streams_labels,
                                      title="Dataset: {} - {} - Evolved_templ: {} - Evolved_thres: {}".format(
                                          dataset_choice, "Isolated", use_evolved_templates, use_evolved_thresholds))
        # plt_creator.plot_gestures(dl.load_dataset(dataset_choice, classes), classes=classes)
    else:
        m_wlcss_cuda = WLCSSCudaContinuous(templates, [stream], 1, use_encoding)
        mss = m_wlcss_cuda.compute_wlcss(np.array([params]))
        m_wlcss_cuda.cuda_freemem()
        plt_creator.plot_continuous_data(stream, labels, timestamps)
        plt_creator.plot_continuous_mss(mss, labels, timestamps, classes, thresholds, peaks=find_peaks(mss),
                                        title="Dataset: {} - {} - Evolved_templ: {} - Evolved_thres: {}".format(
                                            dataset_choice, "Continuous", use_evolved_templates,
                                            use_evolved_thresholds))
        pfe.performance_evaluation_continuous(mss, labels, timestamps, thresholds, classes, wsize=wsize)
    plt.show()
    print("End!")

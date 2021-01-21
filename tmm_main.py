#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from data_processing import data_loader as dl
from performance_evaluation import fitness_functions as ftf
from performance_evaluation import performance_evaluation as pfe
from template_matching.utils import find_peaks
from template_matching.wlcss_cuda_class import WLCSSCudaContinuous, WLCSSCuda
from utils.plots import plot_creator as plt_creator

if __name__ == '__main__':
    dataset_choice = 'beachvolleyball_encoded'
    outputs_path = "/home/mathias/Documents/Academic/PhD/Research/WLCSSTraining/training/cuda"

    isolated_case = True  # True for isolate, False for continuous
    save_img = False
    encoding = False

    split_train_test = True
    train_test_random_state = 42

    use_null = False
    user = None
    use_generated_params = False
    use_evolved_templates = True
    use_evolved_thresholds = False

    print("Dataset: " + dataset_choice)
    print("Isolate: {}".format(isolated_case))
    print("Evolved templates: {}".format(use_evolved_templates))

    write_to_file = True
    if dataset_choice == 'skoda':
        encoding = '3d'
        classes = [3001, 3002, 3003, 3005, 3013, 3014, 3018, 3019]
        output_folder = "{}/skoda/params".format(outputs_path)
        params = [[63, 1, 0], [50, 2, 7], [41, 3, 0], [58, 1, 2], [32, 6, 8], [54, 4, 3], [59, 4, 7], [53, 22, 12]]
        thresholds = [998, 519, -84, 644, -1053, -91, -38, -718]
        null_class_percentage = 0.6
        wsize = 500
        temporal_merging_window = 50
        tolerance_window = 10
        es_results_file = "{}/skoda/variable_templates/zeus_templates_2020-10-08_14-39-47".format(outputs_path)
        if use_evolved_templates:
            thresholds = dl.load_template_generation_thresholds(es_results_file)
    elif dataset_choice == 'skoda_mini':
        encoding = False
        classes = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57]
        output_folder = "{}/skoda_mini/params".format(outputs_path)
        null_class_percentage = 0.6
        params = [[48, 2, 2], [55, 0, 2], [59, 1, 7], [61, 54, 3], [61, 13, 1], [35, 16, 0], [36, 7, 2], [59, 0, 2],
                  [44, 0, 5], [41, 32, 8]]
        thresholds = [1840, 3532, 3737, -2001, 1912, 496, 154, 3205, 1220, -2831]
        es_results_file = "{}/skoda_mini/templates/poseidon_templates_2020-09-11_17-16-46".format(outputs_path)
        if use_evolved_templates:
            thresholds = dl.load_template_generation_thresholds(es_results_file)
        wsize = 1000
        temporal_merging_window = 5
        tolerance_window = 5
    elif dataset_choice == 'opportunity_encoded':
        encoding = '3d'
        classes = [406516, 404516, 406520, 404520, 406505, 404505, 406519, 404519, 408512, 407521, 405506]
        # classes = classes[:-2]
        user = 3
        output_folder = "{}/opportunity_encoded/params".format(outputs_path)
        params = [[579, 7, 7], [906, 35, 12], [996, 72, 2], [947, 114, 9], [965, 165, 9], [619, 12, 14], [985, 148, 2],
                  [918, 30, 1], [1009, 6, 1], [278, 963, 336], [988, 72, 6]]
        thresholds = [7508, 20123, -794, 11293, -34, 10628, 4175, 7202, 10268, 12265, -1133]
        # params = params[:-2]
        es_results_file = "{}/opportunity_encoded/variable_templates/kronos_templates_2020-11-30_14-37-38".format(
            outputs_path)
        if use_evolved_templates:
            thresholds = dl.load_template_generation_thresholds(es_results_file)
    elif dataset_choice == 'hci_guided':
        classes = [49, 50, 51, 52, 53]
        output_folder = "{}/hci_guided/params".format(outputs_path)
        params = [[39, 15, 3], [47, 2, 2], [53, 20, 2], [54, 40, 4], [63, 43, 3]]
        thresholds = [98, 1961, 409, 199, 794]
        wsize = 5
        temporal_merging_window = 5
        null_class_percentage = 0.5
        es_results_file = "{}/hci_guided/all/zeus_all_2020-11-06_15-07-19".format(outputs_path)
        if use_evolved_templates:
            thresholds = dl.load_template_generation_thresholds(es_results_file)
        if use_generated_params:
            params = dl.load_generated_params(es_results_file)
    elif dataset_choice == 'hci_table':
        encoding = '2d'
        classes = [i for i in range(9, 35)]
        output_folder = "{}/hci_table/params".format(outputs_path)
        null_class_percentage = 0.5
        params = [[52, 15, 5], [31, 0, 1], [59, 2, 4], [41, 1, 2], [54, 1, 5], [51, 29, 20], [60, 2, 2], [52, 1, 5],
                  [62, 44, 3], [38, 3, 6], [47, 16, 1], [63, 15, 5], [59, 1, 2], [57, 0, 1], [41, 2, 3], [58, 1, 1],
                  [55, 1, 2], [38, 2, 3], [55, 4, 2], [51, 4, 0], [34, 2, 5], [61, 4, 2], [60, 2, 2], [55, 23, 6],
                  [55, 1, 2], [18, 0, 1]]
        thresholds = [274, 697, 511, 775, 509, -520, 895, 906, -355, 292, -310, 178, 957, 978, 437, 890, 830, 589, 394,
                      433, 333, 679, 927, -372, 688, 264]
        es_results_file = "{}/hci_table/variable_templates/poseidon_templates_2020-10-06_13-10-44".format(outputs_path)
        if use_evolved_templates:
            thresholds = dl.load_template_generation_thresholds(es_results_file)
    elif dataset_choice == 'beachvolleyball':
        classes = [1001, 1002, 1003, 1004]
        output_folder = "{}/beachvolleyball/params_perclass".format(outputs_path)
        params = []
        thresholds = []
    elif dataset_choice == 'beachvolleyball_encoded':
        classes = [1001, 1002, 1003, 1004]
        output_folder = "{}/beachvolleyball_encoded/params_perclass".format(outputs_path)
        encoding = '3d'
        params = [[37, 1, 2], [44, 26, 10], [52, 5, 0], [34, 9, 4]]
        thresholds = [930, -244, 818, -295]
        es_results_file = "{}/beachvolleyball_encoded/variable_templates/zeus_templates_2021-01-20_16-37-08".format(
            outputs_path)
        if use_evolved_templates:
            thresholds = dl.load_template_generation_thresholds(es_results_file)
    elif dataset_choice == 'opportunity':
        classes = [406516, 404516, 406505, 404505, 406519, 404519, 407521, 405506]
        # classes = [406516, 408512, 405506]
        # classes = [407521, 406520, 406505, 406519]
        output_folder = "{}/opportunity/params".format(outputs_path)
        user = 32
        params = [33, 9, 2]
        thresholds = [480, 2036, 921, 2038, 815, 1477, 1797, 0]
        wsize = 500
        es_results_file = "{}/opportunity/templates/templates_2019-04-17_16-33-36".format(outputs_path)
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

    if isolated_case:
        templates, streams, streams_labels = dl.load_training_dataset(dataset_choice=dataset_choice, classes=classes,
                                                                      template_choice_method='mrt_lcs',
                                                                      use_quick_loader=True)
        if split_train_test:
            streams_train, streams_test, train_labels, test_labels = train_test_split(streams, streams_labels,
                                                                                      test_size=.33,
                                                                                      random_state=train_test_random_state,
                                                                                      stratify=streams_labels)
            fig = plt.figure()
            subplt = fig.add_subplot(111)
            subplt.hist(streams_labels, label='Total', align='left')
            subplt.hist(train_labels, label='Training', align='mid')
            subplt.hist(test_labels, label='Test', align='right')
            plt.legend()
            streams_test_labels_sorted_idx = test_labels.argsort()
            streams_test = [streams_test[i] for i in streams_test_labels_sorted_idx]
            test_labels = test_labels[streams_test_labels_sorted_idx]
        else:
            # Group streams by labels
            streams_labels_sorted_idx = streams_labels.argsort()
            streams = [streams[i] for i in streams_labels_sorted_idx]
            streams_labels = streams_labels[streams_labels_sorted_idx]
            streams_test = streams
            streams_train = streams
            train_labels = streams_labels
            test_labels = streams_labels
    else:
        if not use_evolved_templates:
            templates, _, _ = dl.load_training_dataset(dataset_choice=dataset_choice, classes=classes,
                                                       template_choice_method='mrt_lcs')
        stream, labels, timestamps = dl.load_continuous_dataset(dataset_choice=dataset_choice, user=user)

    print("Classes: {}".format(classes))

    if use_evolved_templates:
        if es_results_file is not None:
            if use_evolved_thresholds:
                templates, thresholds = dl.load_evolved_templates(es_results_file, classes,
                                                                  use_evolved_thresholds)
            else:
                templates = dl.load_evolved_templates(es_results_file, classes)

    if isolated_case:
        m_wlcss_cuda = WLCSSCuda(templates, streams_test, params, encoding)
        mss = m_wlcss_cuda.compute_wlcss()
        m_wlcss_cuda.cuda_freemem()
        print("Perf_F1: {}".format(ftf.isolated_fitness_function_params(mss, test_labels, thresholds, classes,
                                                                        parameter_to_optimize='f1')))
        pfe.performance_evaluation_isolated(mss, test_labels, thresholds, classes)
        plt_creator.plot_isolated_mss(mss, thresholds, dataset_choice, classes, streams_labels=test_labels,
                                      title="Dataset: {} - {} - Evolved_templ: {} - Evolved_thres: {}".format(
                                          dataset_choice, "Isolated", use_evolved_templates, use_evolved_thresholds))
        # plt_creator.plot_gestures(dl.load_dataset(dataset_choice, classes), classes=classes)
    else:
        m_wlcss_cuda = WLCSSCudaContinuous(templates, [stream], 1, encoding)
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

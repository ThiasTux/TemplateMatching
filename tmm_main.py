#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np

from data_processing import data_loader as dl
from performance_evaluation import performance_evaluation as pfe
from template_matching.wlcss_cuda_class import WLCSSCudaParamsTraining, WLCSSCudaContinuous
from utils.plots import plot_creator as plt_creator

if __name__ == '__main__':
    dataset_choice = 201

    isolated_case = True  # True for isolate, False for continuous
    save_img = False

    use_null = False
    user = None
    use_evolved_templates = True
    use_evolved_thresholds = True

    write_to_file = True
    if dataset_choice == 100:
        use_encoding = False
        classes = [3001, 3003, 3013, 3018]
        # classes = [3001, 3002, 3003, 3005, 3013, 3014, 3018, 3019]
        output_folder = "outputs/training/cuda/skoda/params"
        sensor = None
        null_class_percentage = 0.6
    elif dataset_choice == 101:
        use_encoding = False
        classes = [3001, 3003, 3013, 3018]
        # classes = [3001, 3002, 3003, 3005, 3013, 3014, 3018, 3019]
        output_folder = "outputs/training/cuda/skoda/params"
        sensor = None
        null_class_percentage = 0.6
        params = [31, 0, 0]
        thresholds = [471, 523, 441, 423]
        es_results_file = "outputs/training/cuda/skoda_old/templates/templates_2019-03-27_18-24-04"
    elif dataset_choice == 200 or dataset_choice == 201 or dataset_choice == 202 or dataset_choice == 203 \
            or dataset_choice == 204 or dataset_choice == 205 or dataset_choice == 211:
        use_encoding = False
        # classes = [406516, 404516, 406520, 404520, 406505, 404505, 406519, 404519, 408512, 407521, 405506]
        # classes = [406516, 408512, 405506]
        classes = [407521, 406520, 406505, 406519]
        output_folder = "outputs/training/cuda/opportunity/params"
        user = 3
        params = [14, 1, 5]
        thresholds = [327, 1021, 636, 505]
        es_results_file = "outputs/training/cuda/opportunity/templates/templates_2019-03-27_18-30-02"
    elif dataset_choice == 300:
        use_encoding = False
        classes = [49, 50, 51, 52, 53]
        output_folder = "outputs/training/cuda/hci_guided/params"
        sensor = 31
        null_class_percentage = 0.5
    elif dataset_choice == 400:
        use_encoding = False
        classes = [49, 50, 51, 52, 53]
        output_folder = "outputs/training/cuda/hci_freehand/params"
        sensor = 52
    elif dataset_choice == 500:
        use_encoding = False
        classes = [0, 7]
        output_folder = "outputs/training/cuda/notmnist/params"
        sensor = 0
        null_class_percentage = 0
    elif dataset_choice == 700:
        use_encoding = False
        classes = [1001, 1002, 1003, 1004]
        output_folder = "outputs/training/cuda/synthetic/params"
        null_class_percentage = 0
        params = [7, 5, 1]
        thresholds = [-2500, -2000, -4000, -2200]
        es_results_file = "outputs/training/cuda/synthetic/templates/templates_2019-03-08_12-04-51"
    elif dataset_choice == 701:
        use_encoding = False
        classes = [1001, 1002]
        output_folder = "outputs/training/cuda/synthetic2/params"
        null_class_percentage = 0
        params = [28, 2, 0]
        thresholds = [991, 567]
        es_results_file = "outputs/training/cuda/synthetic2/templates/templates_2019-03-25_13-31-16"

    if isolated_case:
        chosen_templates, instances, labels = dl.load_training_dataset(dataset_choice=dataset_choice, user=user,
                                                                       classes=classes, extract_null=use_null)
    else:
        if not use_evolved_templates:
            chosen_templates, instances, labels = dl.load_training_dataset(dataset_choice=dataset_choice, user=user,
                                                                           classes=classes, extract_null=use_null)
        stream = dl.load_continuous_dataset(dataset_choice=dataset_choice, user=user, )

    if use_evolved_templates:
        if es_results_file is not None:
            if use_evolved_thresholds:
                chosen_templates, thresholds = dl.load_evolved_templates(es_results_file, classes,
                                                                         use_evolved_thresholds)
            else:
                chosen_templates = dl.load_evolved_templates(es_results_file, classes)

    if isolated_case:
        m_wlcss_cuda = WLCSSCudaParamsTraining(chosen_templates, instances, 1, False)
        mss = m_wlcss_cuda.compute_wlcss(np.array([params]))[0]
        m_wlcss_cuda.cuda_freemem()
        tmp_labels = np.array(labels).reshape((len(instances), 1))
        mss = np.concatenate((mss, tmp_labels), axis=1)
        pfe.performance_evaluation_isolated(mss, thresholds, classes)
        plt_creator.plot_isolated_mss(mss, thresholds, dataset_choice, classes,
                                      title="Dataset: {} - {} - Evolved_templ: {} - Evolved_thres: {}".format(
                                          dataset_choice, "Isolated", use_evolved_templates, use_evolved_thresholds))
        # plt_creator.plot_gestures(dl.load_dataset(dataset_choice, classes), classes=classes)
    else:
        m_wlcss_cuda = WLCSSCudaContinuous(chosen_templates, [stream], 1, False)
        mss = m_wlcss_cuda.compute_wlcss(np.array([params]))
        m_wlcss_cuda.cuda_freemem()
        plt_creator.plot_continuous_data(stream, classes)
        tmp_mss = np.empty((mss.shape[0], mss.shape[1] + 2))
        tmp_mss[:, 0] = stream[:, 0]
        tmp_mss[:, -1] = stream[:, 2]
        tmp_mss[:, 1:-1] = mss
        plt_creator.plot_continuous_mss(tmp_mss, classes, thresholds,
                                        title="Dataset: {} - {} - Evolved_templ: {} - Evolved_thres: {}".format(
                                            dataset_choice, "Continuous", use_evolved_templates,
                                            use_evolved_thresholds))
    plt.show()
    print("End!")

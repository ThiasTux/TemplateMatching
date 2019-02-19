#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np

from data_processing import data_loader as dl
from performance_evaluation import fitness_functions as ftf
from template_matching.wlcss_cuda_class import WLCSSCudaParamsTraining
from utils.plots import plot_creator as plt_creator

if __name__ == '__main__':
    dataset_choice = 201

    stream_modality = 1  # 1 for instances, 2 for complete stream
    save_img = False
    extract_null = False
    eval_templates = False

    use_null = True
    user = None

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
        params = [30, 1, 0]
        thresholds = [337, 195, 304, 232]
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
        thresholds = [-3466, -1576, -15231, -4022]

    chosen_templates, instances, labels = dl.load_training_dataset(dataset_choice=dataset_choice, user=user,
                                                                   classes=classes, extract_null=use_null)

    m_wlcss_cuda = WLCSSCudaParamsTraining(chosen_templates, instances, 1, False)
    mss = m_wlcss_cuda.compute_wlcss(np.array([params]))[0]
    m_wlcss_cuda.cuda_freemem()
    tmp_labels = np.array(labels).reshape((len(instances), 1))
    mss = np.concatenate((mss, tmp_labels), axis=1)
    fitness_score = ftf.isolated_fitness_function_params(mss, thresholds, classes)
    print(fitness_score)
    plt_creator.plot_isolated_mss(mss, thresholds)
    # plt_creator.plot_gestures(dl.load_dataset(dataset_choice, classes), classes=classes)
    plt.show()
    print("End!")

"""
Test WLCSSCuda
"""
import time

import matplotlib.pyplot as plt
import numpy as np

import data_processing.data_loader as dl
import performance_evaluation.fitness_functions as ftf
import utils.plots.plot_creator as plt_creator
from performance_evaluation import performance_evaluation as pfe
from template_matching.wlcss_cuda_class import WLCSSCudaParamsTraining, WLCSSCuda


def simple_wlcss_test():
    templates = [np.array([4, 4, 5, 6, 0, 0, 7, 5, 4, 4])]
    streams = [np.array([4, 4, 5, 6, 0, 0, 7, 5, 4, 4])]
    params = [8, 1, 0]
    m_wlcss_cuda = WLCSSCudaParamsTraining(templates, streams, 1, False)
    mss = m_wlcss_cuda.compute_wlcss(np.array([params]))[0]
    m_wlcss_cuda.cuda_freemem()
    print(mss)


def perclass_wlcss_test():
    # classes = [406516, 404516, 406505, 404505, 406519, 404519, 407521, 405506, 406520, 404520, 408512]
    classes = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57]
    # classes = [406516, 404516, 406505, 404505, 406519, 404519, 407521, 405506]
    templates, streams, streams_labels = dl.load_training_dataset(dataset_choice='skoda_mini', classes=classes,
                                                                  template_choice_method='mrt_lcs')

    streams_labels_sorted_idx = streams_labels.argsort()
    streams = [streams[i] for i in streams_labels_sorted_idx]
    streams_labels = streams_labels[streams_labels_sorted_idx]

    params = [[34, 5, 2], [59, 18, 3], [49, 13, 2], [5, 11, 4], [55, 57, 1], [55, 38, 0], [24, 8, 2], [47, 16, 2],
              [30, 0, 8], [39, 9, 9]]
    # params = [[63, 3, 1] for _ in range(len(classes))]
    thresholds = [500, 994, 907, -698, 962, 405, -643, -74, 859, 480]
    # thresholds = [374, 228, 104, 279, 519, 394, 247, 73]

    start_time = time.time()
    m_wlcss_cuda = WLCSSCuda(templates, streams, np.array(params), False)
    mss = m_wlcss_cuda.compute_wlcss()
    m_wlcss_cuda.cuda_freemem()
    print("Duration: {}".format(time.strftime("%H:%M:%S.%f", time.gmtime(time.time() - start_time))))

    fitness_score = ftf.isolated_fitness_function_params(mss, streams_labels, thresholds, classes,
                                                         parameter_to_optimize='f1')
    print(fitness_score)

    pfe.performance_evaluation_isolated(mss, streams_labels, thresholds, classes)

    plt_creator.plot_isolated_mss(mss, thresholds, 'hci_guided', classes, streams_labels,
                                  title="Isolated matching score - Params opt. - {}".format('hci_guided'))
    plt.show()


if __name__ == '__main__':
    # simple_wlcss_test()
    perclass_wlcss_test()

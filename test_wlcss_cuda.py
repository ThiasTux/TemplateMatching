"""
Test WLCSSCuda
"""
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
    classes = [406516, 404516, 406505, 404505, 406519, 404519, 407521, 405506, 406520, 404520, 408512]
    # classes = [406516, 404516, 406505, 404505, 406519, 404519, 407521, 405506]
    templates, streams, streams_labels = dl.load_training_dataset(dataset_choice='opportunity_encoded', classes=classes,
                                                                  template_choice_method='mrt_lcs')

    streams_labels_sorted_idx = streams_labels.argsort()
    streams = [streams[i] for i in streams_labels_sorted_idx]
    streams_labels = streams_labels[streams_labels_sorted_idx]

    params = [[53, 0, 4], [36, 1, 14], [59, 2, 13], [52, 1, 14], [39, 7, 3], [62, 2, 5], [42, 1, 18], [59, 2, 8],
              [51, 2, 12], [63, 4, 9], [61, 0, 2]]
    # params = [[63, 3, 1] for _ in range(len(classes))]
    thresholds = [801, 849, 1047, 888, 95, 490, 1658, 160, 844, 970, 677]
    # thresholds = [374, 228, 104, 279, 519, 394, 247, 73]

    m_wlcss_cuda = WLCSSCuda(templates, streams, np.array(params), '3d')
    mss = m_wlcss_cuda.compute_wlcss()
    m_wlcss_cuda.cuda_freemem()

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

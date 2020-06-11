#!/usr/bin/env python
import datetime
import time
from os.path import join

import matplotlib.pyplot as plt
import numpy as np

from data_processing import data_loader as dl
from performance_evaluation import fitness_functions as ftf
from template_matching.wlcss_cuda_class import WLCSSCudaParamsTraining
from training.params.ga_params_optimizer import GAParamsOptimizer
from utils.plots import plot_creator as plt_creator
import socket

if __name__ == '__main__':
    dataset_choice = 'skoda'
    outputs_path = "/home/mathias/Documents/Academic/PhD/Research/WLCSSTraining/training/cuda"

    num_test = 1
    use_null = False
    write_to_file = True
    user = None
    use_encoding = False

    num_individuals = 32
    bits_params = 6
    bits_thresholds = 11
    rank = 10
    elitism = 3
    iterations = 1000
    fitness_function = 'f1_acc'
    crossover_probability = 0.3
    mutation_probability = 0.1

    if dataset_choice == 'skoda':
        use_encoding = False
        classes = [3001, 3003, 3013, 3018]
        # classes = [3001, 3002, 3003, 3005, 3013, 3014, 3018, 3019]
        output_folder = "{}/skoda/params".format(outputs_path)
        sensor = None
        null_class_percentage = 0.6
    elif dataset_choice == 'skoda_mini':
        classes = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57]
        output_folder = "{}/skoda_mini/params".format(outputs_path)
        sensor = None
        null_class_percentage = 0.6
    elif dataset_choice == 'skoda_old':
        use_encoding = False
        classes = [3001, 3003, 3013, 3018]
        # classes = [3001, 3002, 3003, 3005, 3013, 3014, 3018, 3019]
        output_folder = "{}/skoda_old/params".format(outputs_path)
        sensor = None
        null_class_percentage = 0.6
    elif dataset_choice == 'opportunity' or dataset_choice == 201 or dataset_choice == 202 or dataset_choice == 203 \
            or dataset_choice == 204 or dataset_choice == 205 or dataset_choice == 211:
        use_encoding = False
        # classes = [406516, 404516, 406520, 404520, 406505, 404505, 406519, 404519, 408512, 407521, 405506]
        classes = [406516, 404516, 406505, 404505, 406519, 404519, 407521, 405506]
        # classes = [406516, 408512, 405506]
        # classes = [407521, 406520, 406505, 406519]
        user = 3
        output_folder = "{}/opportunity/params".format(outputs_path)
        null_class_percentage = 0.5
    elif dataset_choice == 210:
        use_encoding = False
        # classes = [406516, 404516, 406520, 404520, 406505, 404505, 406519, 404519, 408512, 407521, 405506]
        # classes = [406516, 408512, 405506]
        # classes = [407521, 406520, 406505, 406519]
        output_folder = "{}/opportunity/params".format(outputs_path)
        sensor = None
        null_class_percentage = 0.8
    elif dataset_choice == 'hci_guided':
        use_encoding = False
        classes = [49, 50]
        output_folder = "{}/hci_guided/params".format(outputs_path)
        null_class_percentage = 0.5
    elif dataset_choice == 'hci_freehand':
        use_encoding = False
        classes = [49, 50, 51, 52, 53]
        output_folder = "{}/hci_freehand/params".format(outputs_path)
        sensor = 52
    elif dataset_choice == 500:
        use_encoding = False
        classes = [0, 7]
        output_folder = "{}/notmnist/params".format(outputs_path)
        sensor = 0
        null_class_percentage = 0
    elif dataset_choice == 'synthetic1':
        use_encoding = False
        classes = [1001, 1002, 1003, 1004]
        output_folder = "{}/synthetic/params".format(outputs_path)
        null_class_percentage = 0
    elif dataset_choice == 'synthetic2':
        use_encoding = False
        classes = [1001, 1002]
        output_folder = "{}/synthetic2/params".format(outputs_path)
        null_class_percentage = 0
    elif dataset_choice == 'synthetic3':
        classes = [1001, 1002]
        output_folder = "{}/synthetic3/params".format(outputs_path)
        null_class_percentage = 0
    elif dataset_choice == 'synthetic4':
        classes = [1001, 1002, 1003, 1004]
        output_folder = "{}/synthetic4/params".format(outputs_path)
        null_class_percentage = 0
    elif dataset_choice == 'hci_table':
        use_encoding = '2d'
        classes = [i for i in range(1, 5)]
        output_folder = "{}/hci_table/params".format(outputs_path)
        null_class_percentage = 0.5

    templates, streams, streams_labels = dl.load_training_dataset(dataset_choice=dataset_choice, classes=classes)

    # Group streams by labels
    streams_labels_sorted_idx = streams_labels.argsort()
    streams = [streams[i] for i in streams_labels_sorted_idx]
    streams_labels = streams_labels[streams_labels_sorted_idx]

    st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')
    hostname = socket.gethostname().lower()
    print("Dataset choice: {}".format(dataset_choice))
    print("Classes: {}".format(' '.join([str(c) for c in classes])))
    print("Population: {}".format(num_individuals))
    print("Iteration: {}".format(iterations))
    print("Crossover: {}".format(crossover_probability))
    print("Mutation: {}".format(mutation_probability))
    print("Elitism: {}".format(elitism))
    print("Rank: {}".format(rank))
    print("Num tests: {}".format(num_test))
    print("Fitness function: {}".format(fitness_function))
    print("Null class extraction: {}".format(use_null))
    print("Null class percentage: {}".format(null_class_percentage))
    print("Use encoding: {}".format(use_encoding))

    optimizer = GAParamsOptimizer(templates, streams, streams_labels, classes,
                                  use_encoding=use_encoding,
                                  bits_parameters=bits_params,
                                  bits_thresholds=bits_thresholds,
                                  num_individuals=num_individuals, rank=rank,
                                  elitism=elitism,
                                  iterations=iterations,
                                  fitness_function=fitness_function,
                                  cr_p=crossover_probability,
                                  mt_p=mutation_probability)
    start_time = time.time()

    optimizer.optimize()

    elapsed_time = time.time() - start_time
    results = optimizer.get_results()
    output_scores_path = "{}/{}_param_thres_{}_scores.txt".format(output_folder, hostname, st)
    with open(output_scores_path, 'w') as f:
        for item in results[-1]:
            f.write("%s\n" % str(item).replace("[", "").replace("]", ""))
    output_file_path = join(output_folder,
                            "{}_param_thres_{}.txt".format(hostname, st))
    output_config_path = join(output_folder,
                              "{}_param_thres_{}_conf.txt".format(hostname, st))
    with open(output_config_path, 'w') as outputconffile:
        outputconffile.write("Dataset choice: {}\n".format(dataset_choice))
        outputconffile.write("Classes: {}\n".format(' '.join([str(c) for c in classes])))
        outputconffile.write("Population: {}\n".format(num_individuals))
        outputconffile.write("Iteration: {}\n".format(iterations))
        outputconffile.write("Crossover: {}\n".format(crossover_probability))
        outputconffile.write("Mutation: {}\n".format(mutation_probability))
        outputconffile.write("Elitism: {}\n".format(elitism))
        outputconffile.write("Rank: {}\n".format(rank))
        outputconffile.write("Num tests: {}\n".format(num_test))
        outputconffile.write("Fitness function: {}\n".format(fitness_function))
        outputconffile.write("Bit params: {}\n".format(bits_params))
        outputconffile.write("Bit thresholds: {}\n".format(bits_thresholds))
        outputconffile.write("Null class extraction: {}\n".format(use_null))
        outputconffile.write("Null class percentage: {}\n".format(null_class_percentage))
        outputconffile.write("Use encoding: {}\n".format(use_encoding))
        outputconffile.write("Duration: {}\n".format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))

    with open(output_file_path, 'w') as outputfile:
        outputfile.write("{}\n".format(results[:-1]))
    print("Results written")
    print(output_file_path)
    print(results[:-1])

    params = results[0:3]
    thresholds = results[3]

    m_wlcss_cuda = WLCSSCudaParamsTraining(templates, streams, 1, use_encoding)
    mss = m_wlcss_cuda.compute_wlcss(np.array([params]))[0]
    m_wlcss_cuda.cuda_freemem()

    fitness_score = ftf.isolated_fitness_function_params(mss, streams_labels, thresholds, classes,
                                                         parameter_to_optimize='f1')
    print(fitness_score)

    plt_creator.plot_isolated_mss(mss, thresholds, dataset_choice, classes, streams_labels,
                                  title="Isolated matching score - Params opt. - {}".format(dataset_choice))
    plt_creator.plot_scores([output_file_path.replace(".txt", "")],
                            title="Fitness scores evolution - {}".format(dataset_choice))
    plt.show()
    print("End!")

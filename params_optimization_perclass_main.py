#!/usr/bin/env python
import datetime
import socket
import time
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from data_processing import data_loader as dl
from performance_evaluation import fitness_functions as ftf
from template_matching.wlcss_cuda_class import WLCSSCuda
from training.params.ga_params_optimizer import GAParamsOptimizer
from utils.plots import plot_creator as plt_creator

if __name__ == '__main__':
    dataset_choice = 'beachvolleyball_encoded'
    outputs_path = "/home/mathias/Documents/Academic/PhD/Research/WLCSSTraining/training/cuda"

    num_test = 1
    use_null = False
    write_to_file = True
    user = None
    encoding = False
    save_internals = False
    split_train_test = True
    train_test_random_state = 42

    num_individuals = 32
    bits_params = 6
    bits_thresholds = 11
    rank = 10
    elitism = 3
    iterations = 500
    fitness_function = 'f1_acc'
    crossover_probability = 0.35
    mutation_probability = 0.25

    null_class_percentage = 0.5

    if dataset_choice == 'skoda':
        encoding = '3d'
        classes = [3001, 3003, 3013, 3018]
        # classes = [3001, 3002, 3003, 3005, 3013, 3014, 3018, 3019]
        output_folder = "{}/skoda/params_perclass".format(outputs_path)
        null_class_percentage = 0.6
    elif dataset_choice == 'skoda_mini':
        classes = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57]
        output_folder = "{}/skoda_mini/params_perclass".format(outputs_path)
        null_class_percentage = 0.6
    elif dataset_choice == 'opportunity_encoded':
        encoding = '3d'
        classes = [406516, 404516, 406505, 404505, 406519, 404519, 407521, 405506]
        # classes = [408512]
        user = 3
        output_folder = "{}/opportunity_encoded/params_perclass".format(outputs_path)
        null_class_percentage = 0.5
    elif dataset_choice == 'hci_guided':
        classes = [49, 50, 51, 52, 53]
        output_folder = "{}/hci_guided/params_perclass".format(outputs_path)
        null_class_percentage = 0.5
    elif dataset_choice == 'hci_table':
        encoding = '2d'
        classes = [i for i in range(9, 35)]
        output_folder = "{}/hci_table/params_perclass".format(outputs_path)
        null_class_percentage = 0.5
    elif dataset_choice == 'skoda_old':
        classes = [3001, 3003, 3013, 3018]
        # classes = [3001, 3002, 3003, 3005, 3013, 3014, 3018, 3019]
        output_folder = "{}/skoda_old/params_perclass".format(outputs_path)
        null_class_percentage = 0.6
    elif dataset_choice == 'opportunity':
        # classes = [406516, 404516, 406520, 404520, 406505, 404505, 406519, 404519, 408512, 407521, 405506]
        classes = [406516, 404516, 406505, 404505, 406519, 404519, 407521, 405506]
        # classes = [406516, 408512, 405506]
        # classes = [407521, 406520, 406505, 406519]
        user = 3
        output_folder = "{}/opportunity/params_perclass".format(outputs_path)
        null_class_percentage = 0.5
    elif dataset_choice == 'beachvolleyball':
        classes = [1001, 1002, 1003, 1004]
        output_folder = "{}/beachvolleyball/params_perclass".format(outputs_path)
    elif dataset_choice == 'beachvolleyball_encoded':
        classes = [1001, 1002, 1003, 1004]
        output_folder = "{}/beachvolleyball_encoded/params_perclass".format(outputs_path)
        encoding = '3d'
    elif dataset_choice == 'hci_freehand':
        classes = [49, 50, 51, 52, 53]
        output_folder = "{}/hci_freehand/params_perclass".format(outputs_path)
        sensor = 52
    elif dataset_choice == 500:
        classes = [0, 7]
        output_folder = "{}/notmnist/params_perclass".format(outputs_path)
        sensor = 0
        null_class_percentage = 0
    elif dataset_choice == 'synthetic1':
        classes = [1001, 1002, 1003, 1004]
        output_folder = "{}/synthetic/params_perclass".format(outputs_path)
        null_class_percentage = 0
    elif dataset_choice == 'synthetic2':
        classes = [1001, 1002]
        output_folder = "{}/synthetic2/params_perclass".format(outputs_path)
        null_class_percentage = 0
    elif dataset_choice == 'synthetic3':
        classes = [1001, 1002]
        output_folder = "{}/synthetic3/params_perclass".format(outputs_path)
        null_class_percentage = 0
    elif dataset_choice == 'synthetic4':
        classes = [1001, 1002, 1003, 1004]
        output_folder = "{}/synthetic4/params_perclass".format(outputs_path)
        null_class_percentage = 0
    elif dataset_choice == 'shl_preview':
        classes = [1, 2, 4, 7, 8]
        output_folder = "{}/shl_preview/params_perclass".format(outputs_path)
        null_class_percentage = 0.5
    elif dataset_choice == 'uwave_x':
        output_folder = "{}/uwave_x/params_perclass".format(outputs_path)
        classes = [1]
        null_class_percentage = 0.5

    templates, streams, streams_labels = dl.load_training_dataset(dataset_choice=dataset_choice, classes=classes,
                                                                  template_choice_method='mrt_lcs',
                                                                  use_quick_loader=True)

    if split_train_test:
        streams_train, streams_test, train_labels, test_labels = train_test_split(streams, streams_labels,
                                                                                  test_size=.33,
                                                                                  random_state=train_test_random_state)
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
    print("Use encoding: {}".format(encoding))

    results = list()
    output_files = list()

    for i, c in enumerate(classes):
        print(c)
        optimizer = GAParamsOptimizer([templates[i]], streams_train, train_labels, [c],
                                      use_encoding=encoding, save_internals=save_internals,
                                      bits_reward=bits_params,
                                      bits_penalty=bits_params,
                                      bits_epsilon=bits_params,
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
        tmp_results = optimizer.get_results()
        results.append(tmp_results)
        output_file = "{}/{}_param_thres_{}".format(output_folder, hostname, st)
        output_scores_path = "{}_scores_{}.txt".format(output_file, c)
        with open(output_scores_path, 'w') as f:
            for item in tmp_results[-1]:
                f.write("%s\n" % str(item).replace("[", "").replace("]", ""))
        if save_internals:
            output_internal_params_path = "{}_{}_internal_params.csv".format(output_file, c)
            output_internal_scores_path = "{}_{}_internal_scores.csv".format(output_file, c)
            internal_fitness, internal_params = optimizer.get_internal_states()
            np.savetxt(output_internal_scores_path, internal_fitness, fmt='%4.3f')
            np.savetxt(output_internal_params_path, internal_params, fmt='%d')
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
        outputconffile.write("Use encoding: {}\n".format(encoding))
        outputconffile.write("Duration: {}\n".format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))

    params = list()
    thresholds = list()

    print(output_file_path)

    with open(output_file_path, 'w') as outputfile:
        for i, c in enumerate(classes):
            params.append(results[i][0:3])
            thresholds.append(results[i][3])
            outputfile.write("{}\n".format(results[i][:-1]))
            print(c)
            print(results[i][:-1])
    print("Results written")

    m_wlcss_cuda = WLCSSCuda(templates, streams_test, params, encoding)
    mss = m_wlcss_cuda.compute_wlcss()
    m_wlcss_cuda.cuda_freemem()

    fitness_score = ftf.isolated_fitness_function_params(mss, test_labels, thresholds, classes,
                                                         parameter_to_optimize='f1')
    print(fitness_score)

    plt_creator.plot_isolated_mss(mss, thresholds, dataset_choice, classes, test_labels,
                                  title="Isolated matching score - Params opt. - {}".format(dataset_choice))
    plt_creator.plot_perclass_gascores([output_file_path.replace(".txt", "")],
                                       title="Fitness scores evolution - {}".format(dataset_choice))
    print(params)
    print(thresholds)
    plt.show()
    print("End!")

"""
Evaluate Parameters Optimization loading ES parameter from csv file.
"""
import datetime
import socket
import time
from os.path import join

import numpy as np
import pandas as pd

from data_processing import data_loader as dl
from performance_evaluation import fitness_functions as ftf
from template_matching.wlcss_cuda_class import WLCSSCudaParamsTraining
from training.params.ga_params_optimizer import GAParamsOptimizer

test_filepath = "test/params/test_hci_guided_0.csv"
test_info = ["dataset_choice", "num_test", "use_null", "bits_params", "bits_thresholds", "write_to_file", "user",
             "null_class_percentage", "num_individuals", "rank", "elitism", "iterations", "fitness_function",
             "crossover_probability", "mutation_probability", "encoding", "classes", "output_folder"]
print(test_filepath)
test_data = pd.read_csv(test_filepath)

prev_dataset = None

# Load test configuration
for index, td in test_data.iterrows():
    # Load test variables
    for tfi in test_info:
        if td[tfi] == 'None':
            vars()[tfi] = None
        elif td[tfi] == 'FALSE':
            vars()[tfi] = False
        else:
            vars()[tfi] = td[tfi]
    results_paths = list()
    results_scores = list()
    classes = [int(c) for c in classes.replace("[", "").replace("]", "").split(",")]
    timestamp_tests = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')
    # Run tests
    for test in range(num_test):
        # Load data (only if data are not loaded already)
        if prev_dataset is None or dataset_choice != prev_dataset:
            prev_dataset = dataset_choice
            templates, streams, streams_labels = dl.load_training_dataset(dataset_choice=dataset_choice,
                                                                          classes=classes,
                                                                          template_choice_method='mrt_lcs')

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
        print("Bit params: {}".format(bits_params))
        print("Bit thresholds: {}".format(bits_thresholds))
        print("Null class extraction: {}".format(use_null))
        print("Null class percentage: {}".format(null_class_percentage))
        print("Use encoding: {}".format(encoding))

        optimizer = GAParamsOptimizer(templates, streams, streams_labels, classes,
                                      use_encoding=encoding,
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
            outputconffile.write("Use encoding: {}\n".format(encoding))
            outputconffile.write("Duration: {}\n".format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))

        with open(output_file_path, 'w') as outputfile:
            outputfile.write("{}\n".format(results[:-1]))

        params = results[0:3]
        thresholds = results[3]

        m_wlcss_cuda = WLCSSCudaParamsTraining(templates, streams, 1, encoding)
        mss = m_wlcss_cuda.compute_wlcss(np.array([params]))[0]
        m_wlcss_cuda.cuda_freemem()

        fitness_score = ftf.isolated_fitness_function_params(mss, streams_labels, thresholds, classes,
                                                             parameter_to_optimize='f1')
        results_scores.append(fitness_score)
        results_paths.append(output_file_path.replace(".txt", ""))

        test_data.loc[index, 'results_paths'] = str(results_paths)
        test_data.loc[index, 'results_scores'] = str(results_scores)
        test_data.to_csv(test_filepath.replace(".csv", "{}_results.csv".format(timestamp_tests)))
print("Done!")

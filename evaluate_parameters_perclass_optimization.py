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
from template_matching.wlcss_cuda_class import WLCSSCuda
from training.params.ga_params_optimizer import GAParamsOptimizer

test_filepath = "test/params_perclass/test_hci_guided_0.csv"
test_info = ["dataset_choice", "num_test", "use_null", "bits_params", "bits_thresholds", "write_to_file", "user",
             "null_class_percentage", "num_individuals", "rank", "elitism", "iterations", "fitness_function",
             "crossover_probability", "mutation_probability", "encoding", "classes", "output_folder"]
print(test_filepath)
test_data = pd.read_csv(test_filepath)

prev_dataset = None
save_internals = True

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

        results = list()
        output_files = list()

        for i, c in enumerate(classes):
            print(c)
            optimizer = GAParamsOptimizer([templates[i]], streams, streams_labels, [c],
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
                print(results[i][:-1])
        print("Results written")

        m_wlcss_cuda = WLCSSCuda(templates, streams, params, encoding)
        mss = m_wlcss_cuda.compute_wlcss()
        m_wlcss_cuda.cuda_freemem()

        fitness_score = ftf.isolated_fitness_function_params(mss, streams_labels, thresholds, classes,
                                                             parameter_to_optimize='f1')
        results_scores.append(fitness_score)
        results_paths.append(output_file_path.replace(".txt", ""))
        test_data.loc[index, 'results_paths'] = str(results_paths)
        test_data.loc[index, 'results_scores'] = str(results_scores)
        test_data.to_csv(test_filepath.replace(".csv", "{}_results.csv".format(timestamp_tests)))
print("Done!")

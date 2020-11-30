"""
Evaluate Template Generation loading ES parameter from csv file.
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
from training.templates.es_templates_generator import ESVariableTemplateGenerator

test_filepath = "test/variable_templates/test_hci_guided_2.csv"
test_info = ["dataset_choice", "num_test", "use_null", "write_to_file", "user", "params", "thresholds",
             "null_class_percentage", "num_individuals", "rank", "elitism", "iterations", "fitness_function",
             "crossover_probability", "mutation_probability", "inject_templates", "optimize_thresholds", "encoding",
             "classes", "bit_values", "enlarge_probability", "shrink_probability", "length_weight",
             "max_length_rate", "min_length_rate", "output_folder"]

test_data = pd.read_csv(test_filepath)
print(test_filepath)
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
    params = [int(p) for p in params.replace("[", "").replace("]", "").split(",")]
    params = [[params[i], params[i + 1], params[i + 2]] for i in range(0, len(params), 3)]
    thresholds = [int(t) for t in thresholds.replace("[", "").replace("]", "").split(",")]
    timestamp_tests = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')
    # Run tests
    for test in range(num_test):
        # Load data (only if data are not loaded already)
        if prev_dataset is None or dataset_choice != prev_dataset:
            prev_dataset = dataset_choice
            templates, streams, streams_labels = dl.load_training_dataset(dataset_choice=dataset_choice,
                                                                          classes=classes, user=user,
                                                                          extract_null=use_null,
                                                                          template_choice_method='mrt',
                                                                          null_class_percentage=null_class_percentage)
            # Group streams by labels
            streams_labels_sorted_idx = streams_labels.argsort()
            streams = [streams[i] for i in streams_labels_sorted_idx]
            streams_labels = streams_labels[streams_labels_sorted_idx]

        if optimize_thresholds:
            thresholds = [None for _ in range(len(classes))]
            best_thresholds = list()
        st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')
        best_templates = list()
        scores = list()
        start_time = time.time()
        hostname = socket.gethostname().lower()
        print("Dataset choice: {}".format(dataset_choice))
        print("Classes: {}".format(' '.join([str(c) for c in classes])))
        print("Population: {}".format(num_individuals))
        print("Iteration: {}".format(iterations))
        print("Crossover: {}".format(crossover_probability))
        print("Mutation: {}".format(mutation_probability))
        print("Elitism: {}".format(elitism))
        print("Rank: {}".format(rank))
        print("Inject templates: {}".format(inject_templates))
        print("Optimize threshold: {}".format(optimize_thresholds))
        print("Num tests: {}".format(num_test))
        print("Fitness function: {}".format(fitness_function))
        print("Params: {}".format(params))
        print("Thresholds: {}".format(thresholds))
        print("Null class extraction: {}".format(use_null))
        print("Null class percentage: {}".format(null_class_percentage))
        print("Enlarge probability: {}".format(enlarge_probability))
        print("Shrink probability: {}".format(shrink_probability))
        print("Length weight: {}".format(length_weight))
        print("Max length rate: {}".format(max_length_rate))
        print("Min length rate: {}".format(min_length_rate))
        print("Use encoding: {}".format(encoding))
        for i, c in enumerate(classes):
            tmp_labels = np.copy(streams_labels)
            tmp_labels[tmp_labels != c] = 0
            chromosomes = len(templates[i])
            if not inject_templates:
                templates = [None for _ in range(len(classes))]
            print("{} - {}".format(c, chromosomes))
            optimizer = ESVariableTemplateGenerator(streams, tmp_labels, params[i], thresholds[i], c, chromosomes,
                                                    bit_values,
                                                    chosen_template=templates[i], use_encoding=encoding,
                                                    num_individuals=num_individuals, rank=rank,
                                                    elitism=elitism,
                                                    iterations=iterations,
                                                    fitness_function=fitness_function,
                                                    cr_p=crossover_probability,
                                                    mt_p=mutation_probability,
                                                    en_p=enlarge_probability,
                                                    sh_p=shrink_probability,
                                                    l_weight=length_weight,
                                                    max_lr=max_length_rate,
                                                    min_lr=min_length_rate)
            optimizer.optimize()

            results = optimizer.get_results()
            output_file = "{}/{}_templates_{}".format(output_folder, hostname, st)
            output_templates_path = "{}_{}_templates.txt".format(output_file, c)
            output_scores_path = "{}_{}_scores.txt".format(output_file, c)

            if optimize_thresholds:
                best_templates.append(results[1])
                best_thresholds.append(results[2])
                with open(output_scores_path, 'w') as f:
                    for item in results[-3]:
                        f.write("%s\n" % str(item).replace("[", "").replace("]", ""))
                output_templates_path = "{}_{}_templates.txt".format(output_file, c)
                with open(output_templates_path, 'w') as f:
                    for k, item in enumerate(results[-2]):
                        f.write("{} {}\n".format(" ".join([str(x) for x in item.tolist()]),
                                                 results[-1][k]))
            else:
                best_templates.append(results[1])
                thresholds[i] = int((results[2][-1][-2] + results[2][-1][-1]) / 2)
                with open(output_scores_path, 'w') as f:
                    for item in results[-2]:
                        f.write("%s\n" % str(item).replace("[", "").replace("]", ""))
                with open(output_templates_path, 'w') as f:
                    for k, item in enumerate(results[-1]):
                        f.write("{}\n".format(" ".join([str(x) for x in item.tolist()])))

        # best_templates = [np.stack((np.arange(len(r)), r), axis=1) for r in best_templates]
        output_file_path = join(output_folder,
                                "{}_templates_{}.txt".format(hostname, st))
        elapsed_time = time.time() - start_time
        output_config_path = join(output_folder,
                                  "{}_templates_{}_conf.txt".format(hostname, st))
        with open(output_config_path, 'w') as outputconffile:
            outputconffile.write("Dataset choice: {}\n".format(dataset_choice))
            outputconffile.write("Classes: {}\n".format(' '.join([str(c) for c in classes])))
            outputconffile.write("Population: {}\n".format(num_individuals))
            outputconffile.write("Iteration: {}\n".format(iterations))
            outputconffile.write("Crossover: {}\n".format(crossover_probability))
            outputconffile.write("Mutation: {}\n".format(mutation_probability))
            outputconffile.write("Elitism: {}\n".format(elitism))
            outputconffile.write("Rank: {}\n".format(rank))
            outputconffile.write("Inject templates: {}\n".format(inject_templates))
            outputconffile.write("Optimize threshold: {}\n".format(optimize_thresholds))
            outputconffile.write("Num tests: {}\n".format(num_test))
            outputconffile.write("Fitness function: {}\n".format(fitness_function))
            outputconffile.write("Bit values: {}\n".format(bit_values))
            outputconffile.write("Params: {}\n".format(params))
            outputconffile.write("Thresholds: {}\n".format(thresholds))
            outputconffile.write("Null class extraction: {}\n".format(use_null))
            outputconffile.write("Null class percentage: {}\n".format(null_class_percentage))
            outputconffile.write("Enlarge probability: {}\n".format(enlarge_probability))
            outputconffile.write("Shrink probability: {}\n".format(shrink_probability))
            outputconffile.write("Length weight: {}\n".format(length_weight))
            outputconffile.write("Max length rate: {}\n".format(max_length_rate))
            outputconffile.write("Min length rate: {}\n".format(min_length_rate))
            outputconffile.write("Use encoding: {}\n".format(encoding))
            outputconffile.write("Duration: {}\n".format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))
        m_wlcss_cuda = WLCSSCuda(best_templates, streams, params, encoding)
        mss = m_wlcss_cuda.compute_wlcss()
        m_wlcss_cuda.cuda_freemem()

        if optimize_thresholds:
            thresholds = best_thresholds
        fitness_score = ftf.isolated_fitness_function_params(mss, streams_labels, thresholds, classes,
                                                             parameter_to_optimize='f1')
        results_scores.append(fitness_score)
        results_paths.append(output_file_path.replace(".txt", ""))
        test_data.loc[index, 'results_paths'] = str(results_paths)
        test_data.loc[index, 'results_scores'] = str(results_scores)
        test_data.to_csv(test_filepath.replace(".csv", "{}_results.csv".format(timestamp_tests)))
print("Done!")

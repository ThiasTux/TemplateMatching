import datetime
import socket
import time
from os.path import join

import pandas as pd
import numpy as np
from data_processing import data_loader as dl
from performance_evaluation import fitness_functions as ftf
from template_matching.wlcss_cuda_class import WLCSSCudaParamsTraining
from training.templates.es_templates_generator import ESTemplateGenerator, ESTemplateThresholdsGenerator

test_filepath = "test/test_1.csv"
test_info = ["dataset_choice", "num_test", "use_null", "write_to_file", "user", "params", "thresholds",
             "null_class_percentage", "num_individuals", "rank", "elitism", "iterations", "fitness_function",
             "crossover_probability", "mutation_probability", "inject_templates", "optimize_thresholds", "use_encoding",
             "classes", "output_folder", "bit_values", "scaling_length"]

test_data = pd.read_csv(test_filepath)

prev_dataset = None

# Load test configuration
for index, td in test_data.iterrows():
    # Load test variables
    for tfi in test_info:
        if td[tfi] == 'None':
            vars()[tfi] = None
        else:
            vars()[tfi] = td[tfi]
    results_paths = list()
    results_scores = list()
    classes = [int(c) for c in classes.replace("[", "").replace("]", "").split(",")]
    params = [int(p) for p in params.replace("[", "").replace("]", "").split(",")]
    thresholds = [int(t) for t in thresholds.replace("[", "").replace("]", "").split(",")]
    # Run tests
    for test in range(num_test):
        # Load data (only if data are not loaded already)
        if prev_dataset is None or dataset_choice != prev_dataset:
            prev_dataset = dataset_choice
            if inject_templates:
                chosen_templates, instances, labels = dl.load_training_dataset(dataset_choice=dataset_choice,
                                                                               classes=classes, user=user,
                                                                               extract_null=use_null,
                                                                               template_choice_method=1,
                                                                               null_class_percentage=null_class_percentage)
            else:
                instances, labels = dl.load_training_dataset(dataset_choice=dataset_choice,
                                                             classes=classes, user=user, extract_null=use_null,
                                                             template_choice_method=0,
                                                             null_class_percentage=null_class_percentage)
                chosen_templates = [None for _ in range(len(classes))]

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
        for i, c in enumerate(classes):
            tmp_labels = np.copy(labels)
            tmp_labels[tmp_labels != c] = 0
            if inject_templates:
                chromosomes = len(chosen_templates[i])
            else:
                chromosomes = int(
                    np.ceil(np.average([len(t) for t in instances if t[0, -2] == c]).astype(int)) / scaling_length)
            print("{} - {}".format(c, chromosomes))
            if optimize_thresholds:
                optimizer = ESTemplateThresholdsGenerator(instances, tmp_labels, params, c, chromosomes, bit_values,
                                                          file="{}/{}_templates_{}".format(output_folder, hostname, st),
                                                          chosen_template=chosen_templates[i],
                                                          num_individuals=num_individuals, rank=rank,
                                                          elitism=elitism,
                                                          iterations=iterations,
                                                          fitness_function=fitness_function,
                                                          cr_p=crossover_probability,
                                                          mt_p=mutation_probability)
            else:
                optimizer = ESTemplateGenerator(instances, tmp_labels, params, thresholds[i], c, chromosomes,
                                                bit_values,
                                                file="{}/{}_templates_{}".format(output_folder, hostname, st),
                                                chosen_template=chosen_templates[i],
                                                num_individuals=num_individuals, rank=rank,
                                                elitism=elitism,
                                                iterations=iterations,
                                                fitness_function=fitness_function,
                                                cr_p=crossover_probability,
                                                mt_p=mutation_probability)
            optimizer.optimize()

            if optimize_thresholds:
                best_templates += [np.array(r[-2]) for r in optimizer.get_results()]
                best_thresholds += [r[-1] for r in optimizer.get_results()]
            else:
                best_templates += [np.array(r[-1]) for r in optimizer.get_results()]
        best_templates = [np.stack((np.arange(len(r)), r), axis=1) for r in best_templates]
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
            outputconffile.write("Chromosomes: {}\n".format(chromosomes))
            outputconffile.write("Params: {}\n".format(params))
            outputconffile.write("Thresholds: {}\n".format(thresholds))
            outputconffile.write("Null class extraction: {}\n".format(use_null))
            outputconffile.write("Null class percentage: {}\n".format(null_class_percentage))
            outputconffile.write("Duration: {}\n".format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))
        m_wlcss_cuda = WLCSSCudaParamsTraining(best_templates, instances, 1, False)
        mss = m_wlcss_cuda.compute_wlcss(np.array([params]))[0]
        m_wlcss_cuda.cuda_freemem()

        tmp_labels = np.array(labels).reshape((len(instances), 1))
        mss = np.concatenate((mss, tmp_labels), axis=1)
        if optimize_thresholds:
            thresholds = best_thresholds
        fitness_score = ftf.isolated_fitness_function_params(mss, thresholds, classes, parameter_to_optimize=4)
        results_scores.append(fitness_score)
        results_paths.append(output_file_path.replace(".txt", ""))
    test_data.loc[index, 'results_paths'] = str(results_paths)
    test_data.loc[index, 'results_scores'] = str(results_scores)
    test_data.to_csv(test_filepath.replace(".csv", "_results.csv"))
print("Done!")

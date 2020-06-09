#!/usr/bin/env python
"""
Template Generation main launcher
"""
import datetime
import socket
import time
from os.path import join

import matplotlib.pyplot as plt
import numpy as np

from data_processing import data_loader as dl
from performance_evaluation import fitness_functions as ftf
from template_matching.wlcss_cuda_class import WLCSSCudaParamsTraining
from training.templates.es_templates_generator import ESTemplateGenerator, ESTemplateThresholdsGenerator
from utils.plots import plot_creator as plt_creator

if __name__ == '__main__':
    dataset_choice = 'synthetic4'
    outputs_path = "/home/mathias/Documents/Academic/PhD/Research/WLCSSTraining/training/cuda"

    num_test = 1
    use_null = False
    write_to_file = True
    user = None
    params = list()
    thresholds = list()
    null_class_percentage = 0.5

    num_individuals = 256
    rank = 256
    elitism = 5
    iterations = 500
    fitness_function = 86
    crossover_probability = 0.3
    mutation_probability = 0.1
    inject_templates = False
    optimize_thresholds = True

    if dataset_choice == 'skoda':
        use_encoding = '3d'
        classes = [3001, 3003, 3013, 3018]
        # classes = [3001, 3002, 3003, 3005, 3013, 3014, 3018, 3019]
        output_folder = "{}/skoda/params".format(outputs_path)
        null_class_percentage = 0.6
        params = [55, 6, 0]
        thresholds = [186, 104, 195, 115]
        bit_values = 15
    elif dataset_choice == 'skoda_mini':
        use_encoding = False
        classes = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57]
        output_folder = "{}/skoda_mini/params".format(outputs_path)
        null_class_percentage = 0.6
        params = [31, 0, 0]
        thresholds = [471, 523, 441, 423]
        bit_values = 27
    elif dataset_choice == 'skoda_old':
        use_encoding = False
        classes = [3001, 3003, 3013, 3018]
        # classes = [3001, 3002, 3003, 3005, 3013, 3014, 3018, 3019]
        output_folder = "{}/skoda_old/params".format(outputs_path)
        params = [31, 0, 0]
        thresholds = [471, 523, 441, 423]
        bit_values = 27
    elif dataset_choice == 'opportunity' or dataset_choice == 201 or dataset_choice == 202 or dataset_choice == 203 \
            or dataset_choice == 204 or dataset_choice == 205 or dataset_choice == 211:
        use_encoding = False
        classes = [406516, 404516, 406505, 404505, 406519, 404519, 407521, 405506]
        # classes = [406516, 408512, 405506]
        # classes = [407521, 406520, 406505, 406519]
        user = None
        output_folder = "{}/opportunity/params".format(outputs_path)
        null_class_percentage = 0.5
        params = [42, 6, 4]
        thresholds = [460, 979, 968, 1733, 1657, 1784, 1199, 976]
        bit_values = 128
    elif dataset_choice == 210:
        use_encoding = False
        # classes = [406516, 404516, 406520, 404520, 406505, 404505, 406519, 404519, 408512, 407521, 405506]
        # classes = [406516, 408512, 405506]
        # classes = [407521, 406520, 406505, 406519]
        output_folder = "{}/hci_guided/params".format(outputs_path)
        sensor = None
        null_class_percentage = 0.8
    elif dataset_choice == 'hci_guided':
        use_encoding = False
        classes = [49, 50, 51, 52, 53]
        output_folder = "outputs/training/cuda/hci_guided/templates"
        params = [61, 24, 2]
        thresholds = [3, 321, 365, 1024, 1412]
        bit_values = 128
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
        params = [7, 5, 1]
        thresholds = [-3466, -1576, -15231, -4022]
        bit_values = 128
    elif dataset_choice == 'synthetic2':
        use_encoding = False
        classes = [1001, 1002]
        output_folder = "{}/synthetic2/params".format(outputs_path)
        null_class_percentage = 0
        params = [43, 2, 63]
        thresholds = [5886, 4756]
        bit_values = 128
    elif dataset_choice == 'synthetic3':
        use_encoding = False
        classes = [1001, 1002]
        output_folder = "{}/synthetic3/params".format(outputs_path)
        null_class_percentage = 0
        params = [60, 2, 6]
        thresholds = [342, 364]
        bit_values = 64
    elif dataset_choice == 'synthetic4':
        use_encoding = False
        classes = [1001, 1002, 1003, 1004]
        output_folder = "{}/synthetic4/params".format(outputs_path)
        null_class_percentage = 0
        params = [60, 4, 0]
        thresholds = [5534, 165, 3058, 4534]
        bit_values = 64
    elif dataset_choice == 'hci_table':
        use_encoding = '2d'
        classes = [i for i in range(1, 5)]
        output_folder = "{}/hci_table/params".format(outputs_path)
        null_class_percentage = 0.5

    if inject_templates:
        templates, streams, streams_labels = dl.load_training_dataset(dataset_choice=dataset_choice,
                                                                      classes=classes, user=user,
                                                                      extract_null=use_null,
                                                                      template_choice_method='random',
                                                                      null_class_percentage=null_class_percentage)
    else:
        streams, streams_labels = dl.load_training_dataset(dataset_choice=dataset_choice,
                                                           classes=classes, user=user, extract_null=use_null,
                                                           template_choice_method=None,
                                                           null_class_percentage=null_class_percentage)
        templates = [None for _ in range(len(classes))]

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
    for i, c in enumerate(classes):
        tmp_labels = np.copy(streams_labels)
        tmp_labels[tmp_labels != c] = 0
        if inject_templates:
            chromosomes = len(templates[i])
        else:
            chromosomes = int(np.ceil(
                np.average([len(streams[i]) for i, sl in enumerate(streams_labels) if sl == c]).astype(int)) / 5)
        print("{} - {}".format(c, chromosomes))
        if optimize_thresholds:
            optimizer = ESTemplateThresholdsGenerator(streams, tmp_labels, params, c, chromosomes, bit_values,
                                                      chosen_template=templates[i],
                                                      num_individuals=num_individuals, rank=rank,
                                                      elitism=elitism,
                                                      iterations=iterations,
                                                      fitness_function=fitness_function,
                                                      cr_p=crossover_probability,
                                                      mt_p=mutation_probability)
        else:
            optimizer = ESTemplateGenerator(streams, tmp_labels, params, thresholds[i], c, chromosomes, bit_values,
                                            chosen_template=templates[i],
                                            num_individuals=num_individuals, rank=rank,
                                            elitism=elitism,
                                            iterations=iterations,
                                            fitness_function=fitness_function,
                                            cr_p=crossover_probability,
                                            mt_p=mutation_probability)
        optimizer.optimize()

        results = optimizer.get_results()
        output_file = "{}/{}_templates_{}".format(output_folder, hostname, st)
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
            with open(output_scores_path, 'w') as f:
                for item in results[-2]:
                    f.write("%s\n" % str(item).replace("[", "").replace("]", ""))
            output_templates_path = "{}_{}_templates.txt".format(output_file, c)
            with open(output_templates_path, 'w') as f:
                for k, item in enumerate(results[-1]):
                    f.write("{}\n".format(" ".join([str(x) for x in item.tolist()])))

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
    print("Results written")
    print(output_file_path.replace(".txt", ""))

    m_wlcss_cuda = WLCSSCudaParamsTraining(best_templates, streams, 1, False)
    mss = m_wlcss_cuda.compute_wlcss(np.array([params]))[0]
    m_wlcss_cuda.cuda_freemem()

    if optimize_thresholds:
        thresholds = best_thresholds
    fitness_score = ftf.isolated_fitness_function_params(mss, streams_labels, thresholds, classes,
                                                         parameter_to_optimize='f1')
    print(fitness_score)

    plt_creator.plot_templates_scores(output_file_path.replace(".txt", ""))
    plt_creator.plot_isolated_mss(mss, thresholds, dataset_choice, classes, streams_labels,
                                  title="Isolated matching score - Template gen. - {}".format(dataset_choice))
    plt.show()
    print("End!")

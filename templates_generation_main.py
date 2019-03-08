#!/usr/bin/env python
import datetime
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
    dataset_choice = 700

    num_test = 1
    use_null = True
    write_to_file = True
    user = None
    params = list()
    thresholds = list()
    null_class_percentage = 0.6

    num_individuals = 64
    rank = 10
    elitism = 3
    iterations = 500
    fitness_function = 7
    crossover_probability = 0.3
    mutation_probability = 0.1
    inject_templates = True
    optimize_thresholds = False

    if dataset_choice == 100:
        use_encoding = False
        classes = [3001, 3003, 3013, 3018]
        # classes = [3001, 3002, 3003, 3005, 3013, 3014, 3018, 3019]
        output_folder = "outputs/training/cuda/skoda/templates"
        null_class_percentage = 0.6
        params = [31, 0, 0]
        thresholds = [471, 523, 441, 423]
        bit_values = 27
    elif dataset_choice == 101:
        use_encoding = False
        classes = [3001, 3003, 3013, 3018]
        # classes = [3001, 3002, 3003, 3005, 3013, 3014, 3018, 3019]
        output_folder = "outputs/training/cuda/skoda_old/templates"
        params = [31, 0, 0]
        thresholds = [471, 523, 441, 423]
        bit_values = 27
    elif dataset_choice == 200 or dataset_choice == 201 or dataset_choice == 202 or dataset_choice == 203 \
            or dataset_choice == 204 or dataset_choice == 205 or dataset_choice == 211:
        use_encoding = False
        # classes = [406516, 404516, 406520, 404520, 406505, 404505, 406519, 404519, 408512, 407521, 405506]
        # classes = [406516, 408512, 405506]
        classes = [407521, 406520, 406505, 406519]
        user = 3
        output_folder = "outputs/training/cuda/opportunity/templates"
        null_class_percentage = 0.5
        params = [14, 1, 5]
        thresholds = [327, 1021, 636, 505]
        bit_values = 128
    elif dataset_choice == 210:
        use_encoding = False
        # classes = [406516, 404516, 406520, 404520, 406505, 404505, 406519, 404519, 408512, 407521, 405506]
        # classes = [406516, 408512, 405506]
        # classes = [407521, 406520, 406505, 406519]
        output_folder = "outputs/training/cuda/opportunity/templates"
        sensor = None
        null_class_percentage = 0.8
    elif dataset_choice == 300:
        use_encoding = False
        classes = [49, 50, 51, 52, 53]
        output_folder = "outputs/training/cuda/hci_guided/templates"
        sensor = 31
        null_class_percentage = 0.5
    elif dataset_choice == 400:
        use_encoding = False
        classes = [49, 50, 51, 52, 53]
        output_folder = "outputs/training/cuda/hci_freehand/templates"
        sensor = 52
    elif dataset_choice == 500:
        use_encoding = False
        classes = [0, 7]
        output_folder = "outputs/training/cuda/notmnist/templates"
        sensor = 0
        null_class_percentage = 0
    elif dataset_choice == 700:
        use_encoding = False
        classes = [1001, 1002, 1003, 1004]
        output_folder = "outputs/training/cuda/synthetic/templates"
        null_class_percentage = 0
        params = [7, 5, 1]
        thresholds = [-3466, -1576, -15231, -4022]
        bit_values = 128

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
    for i, c in enumerate(classes):
        tmp_labels = np.copy(labels)
        tmp_labels[tmp_labels != c] = 0
        if inject_templates:
            chromosomes = len(chosen_templates[i])
        else:
            chromosomes = int(np.ceil(np.average([len(t) for t in instances if t[0, -2] != 0]).astype(int)))
        if optimize_thresholds:
            optimizer = ESTemplateThresholdsGenerator(instances, tmp_labels, params, c, chromosomes, bit_values,
                                                      file="{}/templates_{}".format(output_folder, st),
                                                      chosen_template=chosen_templates[i],
                                                      num_individuals=num_individuals, rank=rank,
                                                      elitism=elitism,
                                                      iterations=iterations,
                                                      fitness_function=fitness_function,
                                                      cr_p=crossover_probability,
                                                      mt_p=mutation_probability)
        else:
            optimizer = ESTemplateGenerator(instances, tmp_labels, params, thresholds[i], c, chromosomes, bit_values,
                                            file="{}/templates_{}".format(output_folder, st),
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
                            "templates_{}.txt".format(st))
    m_wlcss_cuda = WLCSSCudaParamsTraining(best_templates, instances, 1, False)
    mss = m_wlcss_cuda.compute_wlcss(np.array([params]))[0]
    m_wlcss_cuda.cuda_freemem()

    elapsed_time = time.time() - start_time
    output_config_path = join(output_folder,
                              "templates_{}_conf.txt".format(st))
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
        outputconffile.write("Null class extraction: {}\n".format(use_null))
        outputconffile.write("Null class percentage: {}\n".format(null_class_percentage))
        outputconffile.write("Duration: {}\n".format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))

    tmp_labels = np.array(labels).reshape((len(instances), 1))
    mss = np.concatenate((mss, tmp_labels), axis=1)
    if optimize_thresholds:
        plt_creator.plot_isolated_mss(mss, best_thresholds)
        fitness_score = ftf.isolated_fitness_function_params(mss, best_thresholds, classes)
    else:
        plt_creator.plot_isolated_mss(mss, thresholds)
        fitness_score = ftf.isolated_fitness_function_params(mss, thresholds, classes)
    print(fitness_score)
    print(output_file_path.replace(".txt", ""))
    print("Results written")
    print("End!")
    plt.show()

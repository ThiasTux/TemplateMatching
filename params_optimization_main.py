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

if __name__ == '__main__':
    dataset_choice = 300

    num_test = 1
    use_null = False
    write_to_file = True
    user = None

    num_individuals = 32
    bits_params = 5
    bits_thresholds = 10
    rank = 10
    elitism = 3
    iterations = 200
    fitness_function = 5
    crossover_probability = 0.3
    mutation_probability = 0.1

    if dataset_choice == 100:
        use_encoding = False
        classes = [3001, 3003, 3013, 3018]
        # classes = [3001, 3002, 3003, 3005, 3013, 3014, 3018, 3019]
        output_folder = "outputs/training/cuda/skoda/params"
        sensor = None
        null_class_percentage = 0.6
    elif dataset_choice == 101:
        use_encoding = False
        classes = [3001, 3003, 3013, 3018]
        # classes = [3001, 3002, 3003, 3005, 3013, 3014, 3018, 3019]
        output_folder = "outputs/training/cuda/skoda_old/params"
        sensor = None
        null_class_percentage = 0.6
    elif dataset_choice == 200 or dataset_choice == 201 or dataset_choice == 202 or dataset_choice == 203 \
            or dataset_choice == 204 or dataset_choice == 205 or dataset_choice == 211:
        use_encoding = False
        classes = [406516, 404516, 406520, 404520, 406505, 404505, 406519, 404519, 408512, 407521, 405506]
        # classes = [406516, 408512, 405506]
        # classes = [407521, 406520, 406505, 406519]
        user = 3
        output_folder = "outputs/training/cuda/opportunity/params"
        null_class_percentage = 0.5
    elif dataset_choice == 210:
        use_encoding = False
        # classes = [406516, 404516, 406520, 404520, 406505, 404505, 406519, 404519, 408512, 407521, 405506]
        # classes = [406516, 408512, 405506]
        # classes = [407521, 406520, 406505, 406519]
        output_folder = "outputs/training/cuda/opportunity/params"
        sensor = None
        null_class_percentage = 0.8
    elif dataset_choice == 300:
        use_encoding = False
        classes = [49, 50]
        output_folder = "outputs/training/cuda/hci_guided/params"
        null_class_percentage = 0.5
    elif dataset_choice == 400:
        use_encoding = False
        classes = [49, 50, 51, 52, 53]
        output_folder = "outputs/training/cuda/hci_freehand/params"
        sensor = 52
    elif dataset_choice == 500:
        use_encoding = False
        classes = [0, 7]
        output_folder = "outputs/training/cuda/notmnist/params"
        sensor = 0
        null_class_percentage = 0
    elif dataset_choice == 700:
        use_encoding = False
        classes = [1001, 1002, 1003, 1004]
        output_folder = "outputs/training/cuda/synthetic/params"
        null_class_percentage = 0
    elif dataset_choice == 701:
        use_encoding = False
        classes = [1001, 1002]
        output_folder = "outputs/training/cuda/synthetic2/params"
        null_class_percentage = 0

    chosen_templates, instances, labels = dl.load_training_dataset(dataset_choice=dataset_choice,
                                                                   classes=classes, user=user, extract_null=use_null,
                                                                   null_class_percentage=null_class_percentage)
    st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')

    optimizer = GAParamsOptimizer(chosen_templates, instances, labels, classes,
                                  file="{}/param_thres_{}".format(output_folder, st),
                                  bits_parameters=bits_params,
                                  bits_threshold=bits_thresholds,
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
    output_file_path = join(output_folder,
                            "param_thres_{}.txt".format(st))
    output_config_path = join(output_folder,
                              "param_thres_{}_conf.txt".format(st))
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
        outputconffile.write("Duration: {}\n".format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))

    with open(output_file_path, 'w') as outputfile:
        for t, r in enumerate(results):
            outputfile.write("{} {}\n".format(t, r[0:]))
    print("Results written")
    print(output_file_path)
    print(results[-1][0:])

    params = results[-1][0:3]
    thresholds = results[-1][3]

    m_wlcss_cuda = WLCSSCudaParamsTraining(chosen_templates, instances, 1, False)
    mss = m_wlcss_cuda.compute_wlcss(np.array([params]))[0]
    m_wlcss_cuda.cuda_freemem()

    tmp_labels = np.array(labels).reshape((len(instances), 1))
    mss = np.concatenate((mss, tmp_labels), axis=1)
    plt_creator.plot_isolated_mss(mss, thresholds, dataset_choice, classes,
                                  title="Isolated matching score - Params opt. - {}".format(dataset_choice))
    fitness_score = ftf.isolated_fitness_function_params(mss, thresholds, classes)
    print(fitness_score)
    plt.show()
    print("End!")

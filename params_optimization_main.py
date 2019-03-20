#!/usr/bin/env python
import datetime
import time
from os.path import join

from data_processing import data_loader as dl
from training.params.ga_params_optimizer import GAParamsOptimizer

if __name__ == '__main__':
    dataset_choice = 701

    num_test = 1
    use_null = True
    write_to_file = True
    user = None
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
        # classes = [406516, 404516, 406520, 404520, 406505, 404505, 406519, 404519, 408512, 407521, 405506]
        # classes = [406516, 408512, 405506]
        classes = [407521, 406520, 406505, 406519]
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
        classes = [49, 50, 51, 52, 53]
        output_folder = "outputs/training/cuda/hci_guided/params"
        sensor = 31
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
                                  file="{}/param_thres_{}".format(output_folder, st))
    optimizer.optimize()

    results = optimizer.get_results()
    output_file_path = join(output_folder,
                            "param_thres_{}.txt".format(st))
    output_config_path = join(output_folder,
                              "param_thres_{}_conf.txt".format(st))
    with open(output_file_path, 'w') as outputfile:
        for t, r in enumerate(results):
            outputfile.write("{} {}\n".format(t, r[0:]))
    print(output_file_path)
    print(results[-1][0:])
    print("Results written")
    print("End!")

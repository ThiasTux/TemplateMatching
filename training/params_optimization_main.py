if __name__ == '__main__':
    dataset_choice = 500
    num_test = 1
    if dataset_choice == 100:
        use_encoding = False
        classes = [3001, 3003, 3013, 3018]
        # classes = [3001, 3002, 3003, 3005, 3013, 3014, 3018, 3019]
        output_folder = "outputs/cuda/skoda/params"
        sensor = None
        null_class_percentage = 0.6
    elif dataset_choice == 200 or dataset_choice == 201 or dataset_choice == 202 or dataset_choice == 203 \
            or dataset_choice == 204 or dataset_choice == 205 or dataset_choice == 211:
        use_encoding = False
        classes = [406516, 404516, 406520, 404520, 406505, 404505, 406519, 404519, 408512, 407521, 405506]
        # classes = [406516, 408512, 405506]
        # classes = [407521, 406520, 406505, 406519]
        output_folder = "outputs/cuda/opportunity/params"
        sensor = None
        null_class_percentage = 0.5
    elif dataset_choice == 210:
        use_encoding = False
        # classes = [406516, 404516, 406520, 404520, 406505, 404505, 406519, 404519, 408512, 407521, 405506]
        # classes = [406516, 408512, 405506]
        # classes = [407521, 406520, 406505, 406519]
        output_folder = "outputs/cuda/opportunity/params"
        sensor = None
        null_class_percentage = 0.8
    elif dataset_choice == 300:
        use_encoding = False
        classes = [49, 50, 51, 52, 53]
        output_folder = "outputs/cuda/hci_guided/params"
        sensor = 31
        null_class_percentage = 0.5
    elif dataset_choice == 400:
        use_encoding = False
        classes = [49, 50, 51, 52, 53]
        output_folder = "outputs/cuda/hci_freehand/params"
        sensor = 52
    elif dataset_choice == 500:
        use_encoding = False
        classes = [0, 7]
        output_folder = "outputs/cuda/notmnist/params"
        sensor = 0
        null_class_percentage = 0

    flat_instances, labels, chosen_templates, templates, instances = gdl.load_dataset(dataset_choice=dataset_choice,
                                                                                      isolated=isolated,
                                                                                      classes=classes,
                                                                                      sensor=sensor,
                                                                                      extract_null=use_null,
                                                                                      null_class_percentage=null_class_percentage)

def translate_dataset_name(dataset_name):
    if dataset_name == "synthetic2":
        return 701
    elif dataset_name == "hci_guided":
        return 300
    elif dataset_name == "opportunity":
        return 201
    elif dataset_name == "synthetic3":
        return 702
    elif dataset_name == "synthetic4":
        return 704
    elif dataset_name == "skoda":
        return 100


def extract_params(inputfile):
    with open(inputfile, 'r') as file:
        lines = file.readlines()
    params = [[None, None, None] for _ in range(len(lines))]
    thresholds = [None for _ in range(len(lines))]
    for i, l in enumerate(lines):
        split_values = l.replace("[", "").replace("]", "").split(",")
        params[i][0] = int(split_values[0])
        params[i][1] = int(split_values[1])
        params[i][2] = int(split_values[2])
        thresholds[i] = int(split_values[3])
    print(params)
    print(thresholds)


if __name__ == '__main__':
    extract_params(
        "/home/mathias/Documents/Academic/PhD/Research/WLCSSTraining/training/cuda/hci_table/params_perclass/zeus_param_thres_2020-12-16_19-47-24.txt")

import matplotlib.pyplot as plt

from data_processing import data_loader as dl
from utils.plots import plot_creator as plt_creator

if __name__ == '__main__':
    plot_choice = 1
    input_file = "outputs/training/cuda/opportunity/templates/templates_2019-04-01_19-05-22"
    if plot_choice == 0:
        dataset_choice = 201
        classes = [407521, 406520, 406505, 406519]
        data = dl.load_dataset(dataset_choice, classes)
        plt_creator.plot_gestures(data, classes=classes)
    elif plot_choice == 1:
        dataset_choice = 201
        classes = [407521, 406520, 406505, 406519]
        data = dl.load_continuous_dataset(dataset_choice)
        plt_creator.plot_continuous_data(data, classes=classes)
    elif plot_choice == 2:
        input_files = ["outputs/training/cuda/synthetic2/params/param_thres_2019-03-27_10-50-08"]
        plt_creator.plot_scores(input_files)
    elif plot_choice == 3:
        plt_creator.plot_templates_scores(input_file)
    elif plot_choice == 4:
        plt_creator.plot_templates(input_file)
    elif plot_choice == 5:
        input_path = "/home/mathias/Documents/Datasets/BeachVolleyball/08032019_150000/raw/"
        plt_creator.plot_bluesense_data(input_path, 2)
    plt.show()

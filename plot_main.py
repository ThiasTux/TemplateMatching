import matplotlib.pyplot as plt

from data_processing import data_loader as dl
from utils.plots import plot_creator as plt_creator

if __name__ == '__main__':
    plot_choice = 2
    if plot_choice == 0:
        dataset_choice = 201
        classes = [407521, 406520, 406505, 406519]
        data = dl.load_dataset(dataset_choice, classes, user=3)
        plt_creator.plot_gestures(data, classes=classes)
    elif plot_choice == 1:
        input_files = ["outputs/training/cuda/opportunity/params/param_thres_2019-02-15_15-10-22"]
        plt_creator.plot_scores(input_files)
    elif plot_choice == 2:
        input_files = "outputs/training/cuda/synthetic/templates/templates_2019-03-04_17-03-17"
        plt_creator.plot_templates_scores(input_files)
    elif plot_choice == 3:
        input_path = "/home/mathias/Documents/Datasets/BeachVolleyball/20022019_110000/raw/"
        plt_creator.plot_bluesense_data(input_path, 5)
    plt.show()

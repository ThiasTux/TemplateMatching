import matplotlib.pyplot as plt

from data_processing import data_loader as dl
from utils.plots import plot_creator as plt_creator

if __name__ == '__main__':
    plot_choice = 0
    if plot_choice == 0:
        data = dl.load_dataset(700, [1001, 1002, 1003, 1004])
        plt_creator.plot_gestures(data, classes=[1001, 1002, 1003, 1004])
    elif plot_choice == 1:
        input_files = ["outputs/training/cuda/skoda_old/params/param_thres_2019-02-13_12-23-25"]
        plt_creator.plot_scores(input_files)
    plt.show()

import matplotlib.pyplot as plt

from data_processing import data_loader as dl
from utils.plots import plot_creator as plt_creator

if __name__ == '__main__':
    plot_choice = 0
    if plot_choice == 0:
        data = dl.load_dataset(201, [407521, 406520, 406505, 406519])
        plt_creator.plot_gestures(data, classes=[407521, 406520, 406505, 406519])
    elif plot_choice == 1:
        input_files = ["outputs/training/cuda/opportunity/params/param_thres_2019-02-15_15-10-22"]
        plt_creator.plot_scores(input_files)
    plt.show()

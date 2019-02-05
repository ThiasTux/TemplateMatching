import matplotlib.pyplot as plt

from data_processing import data_loader as dl
from utils.plots import plot_creator as plt_creator

if __name__ == '__main__':
    data = dl.load_dataset(700, [1, 2, 3])
    plt_creator.plot_gestures(data, 0, [1, 2, 3])
    plt.show()

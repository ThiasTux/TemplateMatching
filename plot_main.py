import matplotlib.pyplot as plt

from data_processing import data_loader_old as dl
from utils.plots import plot_creator as plt_creator

if __name__ == '__main__':
    plot_choice = 4
    input_file = "/home/mathias/Documents/Academic/PhD/Research/WLCSSTraining/training/cuda/skoda/params/zeus_templates_2020-06-17_18-28-48"
    dataset_choice = 'synthetic'
    classes = [1001, 1002, 1003, 1004]
    if plot_choice == 0:
        data = dl.load_dataset(dataset_choice, classes)
        plt_creator.plot_gestures(data, classes=classes)
    elif plot_choice == 1:
        data = dl.load_continuous_dataset(dataset_choice)
        plt_creator.plot_continuous_data(data, classes=classes)
    elif plot_choice == 2:
        input_files = [
            "/home/mathias/Documents/Academic/PhD/Research/WLCSSTraining/training/cuda/hci_table/params/zeus_param_thres_2020-06-11_16-02-15"]
        plt_creator.plot_scores(input_files)
    elif plot_choice == 3:
        plt_creator.plot_templates_scores(input_file)
    elif plot_choice == 4:
        plt_creator.plot_templates(input_file)
    elif plot_choice == 5:
        input_path = "/home/mathias/Documents/Datasets/BeachVolleyball/08032019_150000/raw/"
        plt_creator.plot_bluesense_data(input_path, 2)
    elif plot_choice == 6:
        plt_creator.plot_wlcss_heatmap(input_file)
    plt.show()

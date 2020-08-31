import matplotlib.pyplot as plt

from data_processing import data_loader_old as dl
from utils.plots import plot_creator as plt_creator

if __name__ == '__main__':
    plot_choice = 2
    input_file = "/home/mathias/Documents/Academic/PhD/Research/WLCSSTraining/training/cuda/hci_guided/templates/zeus_templates_2020-07-30_14-25-05"
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
            "/home/mathias/Documents/Academic/PhD/Research/WLCSSTraining/training/cuda/hci_guided/params/zeus_param_thres_2020-08-10_15-14-30"]
        plt_creator.plot_gascores(input_files)
    elif plot_choice == 21:
        input_files = [
            '/home/mathias/Documents/Academic/PhD/Research/WLCSSTraining/training/cuda/hci_guided/params/poseidon_param_thres_2020-08-19_14-51-38',
            '/home/mathias/Documents/Academic/PhD/Research/WLCSSTraining/training/cuda/hci_guided/params/poseidon_param_thres_2020-08-19_14-57-07',
            '/home/mathias/Documents/Academic/PhD/Research/WLCSSTraining/training/cuda/hci_guided/params/poseidon_param_thres_2020-08-19_15-02-46',
            '/home/mathias/Documents/Academic/PhD/Research/WLCSSTraining/training/cuda/hci_guided/params/poseidon_param_thres_2020-08-19_15-12-43',
            '/home/mathias/Documents/Academic/PhD/Research/WLCSSTraining/training/cuda/hci_guided/params/poseidon_param_thres_2020-08-19_15-19-31']
        plt_creator.plot_perclass_gascores(input_files)
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

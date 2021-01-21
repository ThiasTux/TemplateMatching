import matplotlib.pyplot as plt

from data_processing import data_loader as dl
from utils.plots import plot_creator as plt_creator

if __name__ == '__main__':
    plot_choice = 41
    input_file = "/home/mathias/Documents/Academic/PhD/Research/WLCSSTraining/training/cuda/beachvolleyball_encoded/variable_templates/zeus_templates_2021-01-20_16-37-08"
    dataset_choice = 'beachvolleyball_encoded'
    classes = [1001, 1002, 1003, 1004]
    if plot_choice == 0:
        templates, streams, streams_labels = dl.load_training_dataset(dataset_choice, classes=classes,
                                                                      use_quick_loader=True)
        plt_creator.plot_gestures(streams, streams_labels, classes=classes)
    elif plot_choice == 1:
        data = dl.load_continuous_dataset(dataset_choice)
        plt_creator.plot_continuous_data(data, classes=classes)
    elif plot_choice == 2:
        input_files = [
            "/home/mathias/Documents/Academic/PhD/Research/WLCSSTraining/training/cuda/hci_guided/params/zeus_param_thres_2020-08-10_15-14-30"]
        plt_creator.plot_gascores(input_files)
    elif plot_choice == 21:
        # input_files = [
        #     '/home/mathias/Documents/Academic/PhD/Research/WLCSSTraining/training/cuda/hci_guided/params/poseidon_param_thres_2020-08-19_14-51-38',
        #     '/home/mathias/Documents/Academic/PhD/Research/WLCSSTraining/training/cuda/hci_guided/params/poseidon_param_thres_2020-08-19_14-57-07',
        #     '/home/mathias/Documents/Academic/PhD/Research/WLCSSTraining/training/cuda/hci_guided/params/poseidon_param_thres_2020-08-19_15-02-46',
        #     '/home/mathias/Documents/Academic/PhD/Research/WLCSSTraining/training/cuda/hci_guided/params/poseidon_param_thres_2020-08-19_15-12-43',
        #     '/home/mathias/Documents/Academic/PhD/Research/WLCSSTraining/training/cuda/hci_guided/params/poseidon_param_thres_2020-08-19_15-19-31']
        input_files = [
            "/home/mathias/Documents/Academic/PhD/Research/WLCSSTraining/training/cuda/opportunity_encoded/params_perclass/poseidon_param_thres_2020-12-08_18-57-59",
            "/home/mathias/Documents/Academic/PhD/Research/WLCSSTraining/training/cuda/opportunity_encoded/params_perclass/poseidon_param_thres_2020-12-20_02-32-42"]
        plt_creator.plot_perclass_gascores(input_files)
    elif plot_choice == 3:
        plt_creator.plot_templates_scores(input_file)
    elif plot_choice == 4:
        plt_creator.plot_templates(input_file)
    elif plot_choice == 41:
        plt_creator.plot_generated_templates_mrt_comparison(input_file)
    elif plot_choice == 5:
        input_path = "/home/mathias/Documents/Datasets/BeachVolleyball/08032019_150000/raw/"
        plt_creator.plot_bluesense_data(input_path, 2)
    elif plot_choice == 6:
        plt_creator.plot_wlcss_heatmap(input_file)
    plt.show()

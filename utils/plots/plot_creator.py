import math

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

COLORS = list(mcolors.BASE_COLORS.keys())


def plot_gestures(data, col, classes=None, use_labels_color=False, save_fig=False):
    """
    Plot signal of multiple gestures, one subplot per gesture

    :param gestures: ndarray, NxM
        set of N gesture with M samples
    :param labels: ndarray, optional
        set of N labels
    :param use_labels_color: boolean, optional
        color each plot differently according to the label
    :param save_fig: boolean
        Print figures to eps file
    """
    for c in classes:
        fig = plt.figure(figsize=(10, 5))
        fig.suptitle("Class: {}".format(c))
        gestures = [d for d in data if d[0, -2] == c]
        num_instances_root = math.sqrt(len(gestures))
        num_rows = (math.floor(num_instances_root) if num_instances_root % math.floor(
            num_instances_root) < 0.5 else math.ceil(num_instances_root))
        num_cols = math.ceil(num_instances_root)
        for i, e in enumerate(gestures):
            sub = fig.add_subplot(num_rows - 1, num_cols + 2, i + 1)
            sub.set_title("{}".format(i))
            instance = e[:, col]
            color = COLORS[int(data[i][0, -1])]
            if use_labels_color:
                color = "k"
            sub.set_xticks([])
            sub.set_xticklabels([])
            # sub.set_yticklabels([])
            sub.set_ylim(0, 100)
            sub.plot(instance, linewidth=1.5, color=color)
        if save_fig:
            fig.savefig(
                "./figures/xy_training_gestures_bad.eps", bbox_inches='tight', format='eps', dpi=1000)

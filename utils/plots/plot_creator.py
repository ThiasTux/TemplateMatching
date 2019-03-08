import glob
import math
import os
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors

from utils import filter_data as fd

COLORS = list(mcolors.BASE_COLORS.keys())
OUTPUT_PATH = "/home/mathias/Documents/Academic/PhD/Publications/2019/ABC/WLCSSLearn/figures"


def plot_gestures(data, col=1, classes=None, use_labels_color=False, save_fig=False):
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
            # sub.set_ylim(-100, 100)
            sub.plot(instance, linewidth=1.5, color=color)
        if save_fig:
            fig.savefig(
                "./figures/xy_training_gestures_bad.eps", bbox_inches='tight', format='eps', dpi=1000)


def plot_scores(input_paths, save_img=False, title=None, output_file=""):
    fig = plt.figure(figsize=(13, 3))
    if title is not None:
        fig.suptitle(title)
    subplt = fig.add_subplot(111)
    for input_path in input_paths:
        scores_files = [file for file in glob.glob(input_path + "*_scores.txt") if os.stat(file).st_size != 0]
        dataset_name = input_path.split("/")[2]
        max_scores = None
        for i in range(len(scores_files)):
            file = scores_files[i]
            scores = np.loadtxt(file, delimiter=",")
            if max_scores is None:
                max_scores = np.zeros([scores.shape[0], len(scores_files)])
            max_scores[:, i] = scores[:, 1]
        mean_scores = np.mean(max_scores, axis=1)
        perc90 = np.percentile(mean_scores, 50)
        perc90_idx = np.argmax(mean_scores >= perc90)
        std_scores = np.std(max_scores, axis=1)
        t = np.arange(len(mean_scores))
        p = subplt.plot(t, mean_scores, label=dataset_name)
        subplt.set_ylabel("Fitness score")
        subplt.set_xlabel("GA Iterations")
        subplt.hlines(y=perc90, xmin=-25, xmax=t[perc90_idx], color='k', linestyle='dashed', zorder=10)
        subplt.axvline(x=t[perc90_idx], ymax=perc90, color='k', linestyle='dashed')
        # y = 0.86 for params, 0.66 for thresholds
        subplt.text(t[perc90_idx] + (0 if dataset_name == "opportunity" else -12), -0.10, "{}".format(t[perc90_idx]),
                    fontsize=14)
        # x = -15 for params
        dt = - 0.01 if dataset_name == "skoda" else 0.07
        subplt.text(-23, perc90 - dt, "{:4.2f}".format(perc90), fontsize=14)
        subplt.fill_between(t, (mean_scores + std_scores / 2), (mean_scores - std_scores / 2),
                            color=lighten_color(p[0].get_color()))
    subplt.set_ylim(0, 1)
    subplt.set_xlim(-25, 520)
    subplt.set_xticks(np.arange(0, 600, 100))
    plt.legend(loc=4)
    if save_img:
        plt.savefig(join(OUTPUT_PATH, output_file), bbox_inches='tight', format='eps', dpi=1200)


def plot_templates_scores(input_path, save_img=False, title=None, output_file=""):
    fig = plt.figure(figsize=(12, 4))
    conf_path = input_path + "_conf.txt"
    with open(conf_path, 'r') as conf_file:
        classes_line_num = 2
        for _ in range(classes_line_num):
            classes_line = conf_file.readline()
        classes = classes_line.split(":")[1].strip().split(" ")
    # fig.suptitle(title.format(len(classes)))
    subplt = fig.add_subplot(111)
    for c in classes:
        scores_files = [file for file in glob.glob(input_path + "*_{}_scores.txt".format(c)) if
                        os.stat(file).st_size != 0]
        max_scores = None
        for i in range(len(scores_files)):
            file = scores_files[i]
            scores = np.loadtxt(file, delimiter=",")
            if max_scores is None:
                max_scores = np.zeros([scores.shape[0], len(scores_files)])
            max_scores[:, i] = scores[:, 1]
        mean_scores = np.mean(max_scores, axis=1)
        std_scores = np.std(max_scores, axis=1)
        t = np.arange(len(mean_scores))
        p_mean = subplt.plot(t, mean_scores, label="{}".format(c))
        perc90 = np.percentile(mean_scores, 50)
        perc90_idx = np.argmax(mean_scores >= perc90)
        p_mean_color = p_mean[0].get_color()
        subplt.hlines(y=perc90, xmin=-10, xmax=t[perc90_idx], color='k', linestyle='dashed', zorder=10)
        subplt.axvline(x=t[perc90_idx], ymax=perc90 / 1000, color='k', linestyle='dashed')
        subplt.text(t[perc90_idx], -0.05, "{}".format(t[perc90_idx]))
        # x = -15 for params
        subplt.text(-10, perc90 + 0.02, "{:4.2f}".format(perc90))
        subplt.fill_between(t, mean_scores + std_scores / 2, mean_scores - std_scores / 2,
                            color=lighten_color(p_mean_color))
    subplt.set_xticks([0, len(mean_scores)])
    subplt.set_ylabel("Score: min(good) - max(bad)")
    subplt.set_xlabel("ES Iterations")
    # subplt.set_ylim(-200, 1100)
    # subplt.set_xlim(-10, 550)
    subplt.set_xticks(np.arange(0, 600, 100))
    plt.legend(loc=4)
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    if save_img:
        plt.savefig(join(OUTPUT_PATH, output_file), bbox_inches='tight', format='eps', dpi=1000)


def plot_templates(input_path, num_templates=20, save_img=False, title=None, output_file=""):
    conf_path = input_path + "_conf.txt"
    with open(conf_path, 'r') as conf_file:
        for i, line in enumerate(conf_file.readlines()):
            if i == 1:
                classes_line = line
            elif i == 3:
                iterations_line = line
            elif i == 10:
                num_test_line = line
        classes = classes_line.split(":")[1].strip().split(" ")
        iterations = int(iterations_line.split(":")[1].strip())
        num_test = int(num_test_line.split(":")[1].strip())
    if num_templates < iterations:
        templates_reduction_factor = (iterations / num_templates)
    else:
        templates_reduction_factor = 1
    num_instances_root = math.sqrt(20)
    num_rows = (math.floor(num_instances_root) if num_instances_root % math.floor(
        num_instances_root) < 0.5 else math.ceil(num_instances_root))
    num_cols = math.ceil(num_instances_root)
    for c in classes:
        for t in range(num_test):
            templates_file_path = input_path + ("_{:02d}_{}_templates.txt".format(t, c))
            fig = plt.figure()
            fig.suptitle("{} - {}".format(c, t))
            with open(templates_file_path, 'r') as templates_file:
                j = 1
                for i, line in enumerate(templates_file.readlines()):
                    if i % templates_reduction_factor == 0:
                        t = [int(v) for v in line.split(" ")[:-1]]
                        subplt = fig.add_subplot(num_rows - 1, num_cols + 2, j)
                        subplt.plot(t)
                        subplt.set_title("{}".format(i))
                        j += 1


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('limegreen', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def plot_isolated_mss(mss, thresholds):
    num_instances = mss.shape[0]
    num_templates = mss.shape[1] - 1
    if len(thresholds) != num_templates:
        print("Not enough thresholds!")
        return None
    fig = plt.figure()
    for t in range(num_templates):
        subplt = fig.add_subplot(num_templates, 1, t + 1)
        subplt.plot(mss[:, t], '.', markersize=3)
        subplt.axes.axhline(thresholds[t], color='orange')


def plot_bluesense_data(input_path, channel):
    fig = plt.figure()
    files = [file for file in glob.glob(input_path + "*.LOG") if os.stat(file).st_size != 0]
    shared_xaxis = None
    for i, file in enumerate(files):
        sensor_name = file.split("/")[-1].split(".")[0]
        data = np.loadtxt(file)
        data_time = data[:, 1]
        freq = 500
        cut_off = 5
        x_data = fd.butter_lowpass_filter(data[:, channel], cut_off, freq)
        y_data = fd.butter_lowpass_filter(data[:, channel + 1], cut_off, freq)
        z_data = fd.butter_lowpass_filter(data[:, channel + 2], cut_off, freq)
        if shared_xaxis is None:
            subplt = fig.add_subplot(len(files), 1, i + 1)
            shared_xaxis = subplt
        else:
            subplt = fig.add_subplot(len(files), 1, i + 1, sharex=shared_xaxis)
        subplt.set_title("{}".format(sensor_name))
        subplt.plot(data_time, x_data, data_time, y_data, data_time, z_data, linewidth=0.5)

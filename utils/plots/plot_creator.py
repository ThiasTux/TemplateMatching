import glob
import os
from os.path import join

import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors, patches
from template_matching import wlcss_c as wlcss
from data_processing import data_loader_old as dl
from utils import utils

from utils import filter_data as fd

COLORS = list(mcolors.BASE_COLORS.keys())
OUTPUT_PATH = "/home/mathias/Documents/Academic/PhD/Publications/2019/ABC/WLCSSLearn/figures"


def plot_gestures_old(data, col=1, classes=None, use_labels_color=False, save_fig=False):
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
            # sub.set_xticks([])
            # sub.set_xticklabels([])
            # sub.set_yticklabels([])
            # sub.set_ylim(-100, 100)
            sub.plot(instance, linewidth=1.5, color=color)
        if save_fig:
            fig.savefig(
                "./figures/xy_training_gestures_bad.eps", bbox_inches='tight', format='eps', dpi=1000)


def plot_gestures(data, labels, classes=None, use_labels_color=False, save_fig=False):
    if classes is None:
        classes = np.unique(labels)
    for c in classes:
        fig = plt.figure(figsize=(10, 5))
        fig.suptitle("Class: {}".format(c))
        fig.canvas.set_window_title("{}".format(c))
        gestures_idx = np.where(labels == c)[0]
        gesture_data = [data[d] for d in gestures_idx]
        num_instances_root = math.sqrt(len(gesture_data))
        num_rows = (math.floor(num_instances_root) if num_instances_root % math.floor(
            num_instances_root) < 0.5 else math.ceil(num_instances_root))
        num_cols = math.ceil(num_instances_root)
        for i, e in enumerate(gesture_data):
            sub = fig.add_subplot(num_rows, num_cols + 2, i + 1)
            sub.set_title("{}".format(i))
            # sub.set_xticks([])
            # sub.set_xticklabels([])
            # sub.set_yticks([i for i in range(9)])
            # sub.set_ylim(-1, 9)
            sub.plot(e)


def plot_gascores(input_paths, save_img=False, title=None, output_file=""):
    fig = plt.figure(figsize=(13, 3))
    if title is not None:
        fig.suptitle(title)
    else:
        fig.suptitle("Scores")
    subplt = fig.add_subplot(111)
    for input_path in input_paths:
        with open("{}_conf.txt".format(input_path)) as conf_file:
            conf_file.readline()
            conf_file.readline()
            conf_file.readline()
            iterations = int(conf_file.readline().split(":")[1].strip())
        scores_files = [file for file in glob.glob(input_path + "*_scores.txt") if os.stat(file).st_size != 0]
        dataset_name = input_path.split("/")[10]
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
        p = subplt.plot(t, mean_scores, label=dataset_name)
        subplt.set_ylabel("Fitness score")
        subplt.set_xlabel("GA Iterations")
        subplt.fill_between(t, (mean_scores + std_scores / 2), (mean_scores - std_scores / 2),
                            color=lighten_color(p[0].get_color()))
    subplt.set_ylim(0, 1.1)
    subplt.set_xlim(-10, iterations + int(iterations * 0.1))
    subplt.set_xticks(np.arange(0, iterations + int(iterations * 0.1), int(iterations / 5)))
    plt.legend(loc=4)
    if save_img:
        plt.savefig(join(OUTPUT_PATH, output_file), bbox_inches='tight', format='eps', dpi=1200)


def plot_perclass_gascores(input_paths, save_img=False, title=None, output_file=""):
    fig = plt.figure(figsize=(13, 3))
    if title is not None:
        fig.suptitle(title)
    else:
        fig.suptitle("Scores")
    subplt = fig.add_subplot(111)
    for input_path in input_paths:
        with open("{}_conf.txt".format(input_path)) as conf_file:
            conf_file.readline()
            classes = [int(i) for i in conf_file.readline().split(":")[1].strip().split(" ")]
            conf_file.readline()
            iterations = int(conf_file.readline().split(":")[1].strip())
        for i, c in enumerate(classes):
            scores_files = [file for file in glob.glob(input_path + "*_scores_{}.txt".format(c)) if
                            os.stat(file).st_size != 0]
            dataset_name = input_path.split("/")[10]
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
            p = subplt.plot(t, mean_scores, label=c)
            perc90 = np.percentile(mean_scores, 90)
            perc90_idx = np.argmax(mean_scores >= perc90)
            annot_max(perc90_idx, perc90, subplt)
            subplt.fill_between(t, (mean_scores + std_scores / 2), (mean_scores - std_scores / 2),
                                color=lighten_color(p[0].get_color()))
    subplt.set_ylim(0, 1.1)
    subplt.set_xlim(-10, iterations + int(iterations * 0.1))
    subplt.set_xticks(np.arange(0, iterations + int(iterations * 0.1), int(iterations / 5)))
    plt.legend(loc=4)
    if save_img:
        plt.savefig(join(OUTPUT_PATH, output_file), bbox_inches='tight', format='eps', dpi=1200)


def annot_max(xmax, ymax, ax=None):
    text = "x={:.3f}, y={:.3f}".format(xmax, ymax)
    if not ax:
        ax = plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops = dict(arrowstyle="->", connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data',
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), **kw)


def plot_templates_scores(input_path, save_img=False, title=None, output_file=""):
    fig = plt.figure(figsize=(12, 4))
    conf_path = input_path + "_conf.txt"
    with open(conf_path, 'r') as conf_file:
        classes_line_num = 2
        for _ in range(classes_line_num):
            classes_line = conf_file.readline()
        classes = classes_line.split(":")[1].strip().split(" ")
    fig.suptitle("Template scores: {}".format(input_path))
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
        max_value = np.max(mean_scores)
        max_value_idx = np.argmax(mean_scores >= max_value)
        p_mean_color = p_mean[0].get_color()
        subplt.hlines(y=max_value, xmin=-10, xmax=t[max_value_idx], color='k', linestyle='dashed', zorder=10)
        subplt.axvline(x=t[max_value_idx], ymax=max_value / 1000, color='k', linestyle='dashed')
        subplt.text(t[max_value_idx], -0.5, "{}".format(t[max_value_idx]))
        # x = -15 for params
        subplt.text(-10, max_value + 0.02, "{:4.2f}".format(max_value))
        subplt.fill_between(t, mean_scores + std_scores / 2, mean_scores - std_scores / 2,
                            color=lighten_color(p_mean_color))
    subplt.set_ylabel("Score")
    subplt.set_xlabel("ES Iterations")
    subplt.set_xticks(np.arange(0, len(mean_scores) + 1, 100))
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
        templates_file_path = input_path + ("_{}_templates.txt".format(c))
        fig = plt.figure()
        fig.suptitle("{}".format(c))
        with open(templates_file_path, 'r') as templates_file:
            j = 1
            for i, line in enumerate(templates_file.readlines()):
                if i % templates_reduction_factor == 0:
                    t = [int(v) for v in line.split(" ")[:-1]]
                    subplt = fig.add_subplot(num_rows - 1, num_cols + 2, j)
                    subplt.plot(t, linewidth=.5)
                    subplt.set_title("{}".format(i))
                    # subplt.set_yticklabels([])
                    # subplt.set_xticklabels([])
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


def plot_isolated_mss(mss, thresholds, dataset_choice, classes, streams_labels=None, title=None):
    num_instances = mss.shape[0]
    num_templates = mss.shape[1]
    stream_labels_values = sorted(np.unique(streams_labels))
    color_stream_labels = [stream_labels_values.index(s) for s in streams_labels]
    if len(thresholds) != num_templates:
        print("Not enough thresholds!")
        return None
    fig = plt.figure()
    if title is None:
        fig.suptitle("Isolated matching scores - {}".format(dataset_choice))
    else:
        fig.suptitle(title)
    x = np.arange(len(mss))
    for t in range(num_templates):
        subplt = fig.add_subplot(num_templates, 1, t + 1)
        if streams_labels is None:
            subplt.scatter(x, mss[:, t], s=3)
        else:
            subplt.scatter(x, mss[:, t], c=color_stream_labels, s=10, cmap=plt.get_cmap('tab20'))
        subplt.set_title("{}".format(classes[t]))
        subplt.axes.axhline(thresholds[t], color='orange')


def plot_bluesense_data(input_path, channel):
    users = ["user1", "user2", "user3", "user4"]
    for user in users:
        fig = plt.figure()
        files = [file for file in glob.glob(join(input_path, user) + "/hand*.txt") if os.stat(file).st_size != 0]
        shared_xaxis = None
        for i, file in enumerate(files):
            sensor_name = file.split("/")[-1].split(".")[0]
            data = np.loadtxt(file)
            data_time = data[:, 0]
            freq = 500
            cut_off = 0
            if cut_off != 0:
                x_data = fd.butter_lowpass_filter(data[:, channel], cut_off, freq)
                y_data = fd.butter_lowpass_filter(data[:, channel + 1], cut_off, freq)
                z_data = fd.butter_lowpass_filter(data[:, channel + 2], cut_off, freq)
            else:
                x_data = data[:, channel]
                y_data = data[:, channel + 1]
                z_data = data[:, channel + 2]
            if shared_xaxis is None:
                subplt = fig.add_subplot(len(files), 1, i + 1)
                shared_xaxis = subplt
            else:
                subplt = fig.add_subplot(len(files), 1, i + 1, sharex=shared_xaxis)
            subplt.set_title("{} - Cut_off {}".format(sensor_name, cut_off))
            subplt.plot(data_time, x_data, data_time, y_data, data_time, z_data, linewidth=0.5)
            # subplt.plot(data_time, x_data[::50], linewidth=0.5)


def plot_continuous_data(data, labels, timestamps, save_fig=False):
    fig = plt.figure()
    classes = sorted(np.unique(labels))
    if classes[0] == 0:
        classes[0] = classes[1] - 1
        labels_null = np.where(labels == 0)[0]
        labels[labels_null] = classes[1] - 1
    data_subplt = fig.add_subplot(211)
    labels_subplt = fig.add_subplot(212, sharex=data_subplt)
    data_subplt.plot(timestamps, data, linewidth=0.5)
    labels_subplt.plot(timestamps, labels, '.')


def plot_continuous_mss(mss, labels, timestamps, classes, thresholds, peaks=None, title=""):
    fig = plt.figure()
    fig.suptitle(title)
    num_subplt = len(classes)
    for i, c in enumerate(classes):
        tmp_labels = np.copy(labels)
        tmp_labels[tmp_labels != c] = 0
        tmp_labels_diffs = tmp_labels[1:] - tmp_labels[0:-1]
        start_idx = np.where(tmp_labels_diffs > 0)[0]
        end_idx = np.where(tmp_labels_diffs < 0)[0]
        subplt = fig.add_subplot(num_subplt, 1, i + 1)
        subplt.set_title("{}".format(c))
        subplt.plot(timestamps, mss[:, i], linewidth=0.5)
        min_mss = min(mss[:, i])
        max_mss = max(mss[:, i])
        subplt.axes.axhline(thresholds[i], color='orange')
        for start_act, end_act in zip(start_idx, end_idx):
            subplt.add_patch(
                patches.Rectangle(
                    (timestamps[start_act], min_mss),
                    timestamps[end_act] - timestamps[start_act],
                    max_mss - min_mss,
                    facecolor="y",
                    alpha=0.5,
                    zorder=10
                )
            )
        if peaks is not None:
            y = mss[peaks[i], i]
            x = timestamps[peaks[i]]
            subplt.plot(x, y, 'x', color='r', markersize=5)


def plot_wlcss_heatmap(input_path, templates=None):
    conf_path = input_path + "_conf.txt"
    dataset_name = input_path.split("/")[3]
    with open(conf_path, 'r') as conf_file:
        for i, line in enumerate(conf_file.readlines()):
            if i == 1:
                classes_line = line
            elif i == 13:
                params_line = line
    classes = [int(c) for c in classes_line.split(":")[1].strip().split(" ")]
    params = [int(x) for x in params_line.split(":")[1].strip().strip('][').split(', ')]
    dataset_choice = utils.translate_dataset_name(dataset_name)
    instances = dl.load_dataset(dataset_choice, classes)
    mss = [[None for _ in instances] for _ in classes]
    min_mss = None
    max_mss = None
    for j, c in enumerate(classes):
        template_input_path = input_path + "_00_{}_templates.txt".format(c)
        t_data = np.loadtxt(template_input_path)
        template = t_data[-1, 0:-1]
        for k, i in enumerate(instances):
            _, mss[j][k] = wlcss.compute_wlcss(template, i[:, 1], params[1], params[0], params[2])
            if min_mss is None or min_mss > np.min(mss[j][k]):
                min_mss = np.min(mss[j][k])
            if max_mss is None or max_mss < np.max(mss[j][k]):
                max_mss = np.max(mss[j][k])
    for j, c in enumerate(classes):
        fig = plt.figure()
        fig.suptitle("{} - {}".format(dataset_name, c))
        num_inst_label_root = math.sqrt(len(instances))
        num_rows = math.floor(num_inst_label_root) if num_inst_label_root % math.floor(
            num_inst_label_root) < 0.5 else math.ceil(num_inst_label_root)
        num_cols = math.ceil(num_inst_label_root)
        for k, i in enumerate(instances):
            sub = fig.add_subplot(num_rows, num_cols, k + 1)
            sub.imshow(mss[j][k][1:, 1:], vmin=min_mss, vmax=max_mss)
            sub.set_title("{} - {}".format(int(i[0, 2]), mss[j][k][-1, -1]))

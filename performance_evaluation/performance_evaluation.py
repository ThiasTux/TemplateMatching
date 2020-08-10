import functools
import itertools

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score

colors_key = ['blue', 'red', 'green', 'orange', 'magenta', 'brown', 'cyan', 'yellow', 'steelblue', 'crimson',
              'mediumspringgreen']


def performance_evaluation_isolated(matching_scores, streams_labels, thresholds, classes, compute_conf_matrix=True,
                                    print_results=True):
    num_classes = len(classes)
    num_instances = matching_scores.shape[0]
    true_positive = np.zeros([num_classes])
    false_positive = np.zeros([num_classes])
    true_negative = np.zeros([num_classes])
    false_negative = np.zeros([num_classes])
    for j in range(num_classes):
        for i in range(num_instances):
            act = classes[j]
            label = streams_labels[i]
            test_matching_scores = matching_scores[i, j]
            if test_matching_scores >= thresholds[j] and act == label:
                true_positive[j] += 1
            elif test_matching_scores < thresholds[j] and act == label:
                false_negative[j] += 1
            elif test_matching_scores >= thresholds[j] and act != label:
                false_positive[j] += 1
            elif test_matching_scores < thresholds[j] and act != label:
                true_negative[j] += 1
    accuracy = [(true_positive[k] + true_negative[k]) / num_instances for k in range(num_classes)]
    accuracy.append((np.sum(true_positive) + np.sum(true_negative)) / (num_classes * num_instances))
    precision = [true_positive[k] / (true_positive[k] + false_positive[k]) for k in range(num_classes)]
    precision.append(np.sum(true_positive) / (np.sum(true_positive) + np.sum(false_positive)))
    recall = [true_positive[k] / (true_positive[k] + false_negative[k]) for k in range(num_classes)]
    recall.append(np.sum(true_positive) / (np.sum(true_positive) + np.sum(false_negative)))
    f1 = [2 / (1 / recall[k] + 1 / precision[k]) for k in range(num_classes)]
    f1.append(2 / (1 / recall[num_classes] + 1 / precision[num_classes]))
    if compute_conf_matrix:
        cfm = compute_confusion_matrix_isolated(matching_scores, streams_labels, thresholds, classes)
    else:
        cfm = None
    if print_results:
        print("Accuracy - {}".format(accuracy))
        print("Precision - {}".format(precision))
        print("Recall - {}".format(recall))
        print("F1 - {}".format(f1))
    return [accuracy, precision, recall, f1, cfm]


def compute_confusion_matrix_isolated(matching_scores, streams_labels, thresholds, classes, save_fig=False):
    tmp_streams_labels = np.copy(streams_labels)
    for i in range(len(tmp_streams_labels)):
        if not (tmp_streams_labels[i] in classes):
            tmp_streams_labels[i] = 0
    scores = matching_scores - thresholds
    # perc_scores = minmax_scale(scores, axis=1)
    perc_scores_clip = scores.clip(min=0)
    labels = np.zeros(len(perc_scores_clip))
    for i in range(len(perc_scores_clip)):
        if sum(perc_scores_clip[i]) > 0:
            labels[i] = classes[np.argmax(perc_scores_clip[i])]
    # labels = np.array([classes[i] for i in np.argmax(perc_scores, axis=1)])
    print("Acc: {}".format(accuracy_score(tmp_streams_labels, labels)))
    print("Prec: {}".format(precision_score(tmp_streams_labels, labels, average='macro')))
    print("Rec: {}".format(recall_score(tmp_streams_labels, labels, average='macro')))
    print("F1: {}".format(f1_score(tmp_streams_labels, labels, average='macro')))
    classes.append(0)
    cfm = confusion_matrix(tmp_streams_labels, labels.astype(int), labels=classes, normalize='true')
    plt.figure()
    plt.imshow(cfm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix - Isolated ")
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    fmt = '.2f'
    thresh = cfm.max() / 2.
    for i, j in itertools.product(range(cfm.shape[0]), range(cfm.shape[1])):
        plt.text(j, i, format(cfm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cfm[i, j] > thresh else "black")
    if save_fig:
        plt.savefig(
            "/home/mathias/Documents/Academic/PhD/Publications/2018/ISWC/OptimizationWLCSS/figures/conf_matrix.eps",
            bbox_inches='tight', format='eps', dpi=1000)
    return cfm


def performance_evaluation_continuous(matching_scores, labels, timestamps, thresholds, classes, wsize=500,
                                      temporal_merging_window=5, tolerance_window=5, compute_conf_matrix=True,
                                      print_results=True):
    multiple_matches = 0
    w_overlap = 1
    predicted_labels = np.array([], dtype=int)
    actual_labels = np.array([], dtype=int)
    time = np.array([])
    start_time = timestamps[0]
    end_time = timestamps[-1]
    matching_scores = (matching_scores - thresholds) / thresholds
    for i in np.arange(start_time, end_time, int(wsize * w_overlap)):
        end_w = i + wsize
        x_scores = matching_scores[np.where((timestamps > i) & (timestamps < end_w))[0]]
        if len(x_scores) > 0:
            x_label = labels[np.where((timestamps > i) & (timestamps < end_w))[0]]
            x_times = timestamps[np.where((timestamps > i) & (timestamps < end_w))[0]]
            time = np.append(time, x_times[-1])
            actual_labels = np.append(actual_labels, x_label[-1])
            tmp = x_scores
            tmp[tmp < 0] = 0
            accumulated_wscores = np.trapz(tmp, x_times - x_times[0], axis=0)
            if np.sum(accumulated_wscores) > 0:
                pred_class = classes[np.argmax(accumulated_wscores)]
                multiple_matches += np.count_nonzero(accumulated_wscores == np.max(accumulated_wscores)) - 1
            else:
                pred_class = 0
            predicted_labels = np.append(predicted_labels, pred_class)
    prediction_array = np.stack((time, predicted_labels, actual_labels), axis=1)
    test_events_dict, ground_events_dict = event_extraction(prediction_array, classes)
    if print_results:
        for c in classes:
            print("{} \nDetected events: {} \nGround truth events: {}".format(c, len(test_events_dict[c]),
                                                                              len(ground_events_dict[c])))
        print("Multiple matches: {}".format(multiple_matches))
    results = mat_eval(test_events_dict, ground_events_dict, classes, temporal_merging_window, tolerance_window,
                       print_results, compute_conf_matrix)
    return results


def mat_eval(detected_events_dict, ground_truth_events_dict, classes, temporal_merging_window, tolerance_window,
             print_results=True, compute_conf_matrix=True):
    detected_events_list = [(j[0], j[1], k) for k in detected_events_dict.keys() for j in detected_events_dict[k]]
    detected_events_list.sort(key=lambda tup: tup[0])
    ground_truth_events_list = [(j[0], j[1], k) for k in ground_truth_events_dict.keys() for j in
                                ground_truth_events_dict[k]]
    ground_truth_events_list.sort(key=lambda tup: tup[0])

    if print_results:
        fig = plt.figure(figsize=(10, 3))
        colors = matplotlib.colors.cnames
        for gte in ground_truth_events_list:
            if gte[2] != 0:
                color = colors[colors_key[classes.index(gte[2])]]
                plt.axvspan(gte[0], gte[1], 0.66, 1, color=color)
        for dte in detected_events_list:
            color = colors[colors_key[classes.index(dte[2])]]
            plt.axvspan(dte[0], dte[1], 0.33, 0.66, color=color)

    # Temporal filter for merging close samples
    m_detected_events_list = detected_events_list
    i = 0
    merged_events = 0

    while i < len(m_detected_events_list) - 1:
        current_event = m_detected_events_list[i]
        next_event = m_detected_events_list[i + 1]
        if (next_event[0] - current_event[1]) <= temporal_merging_window and next_event[2] == current_event[2]:
            merged_events += 1
            merged_event = (current_event[0], next_event[1], current_event[2])
            m_detected_events_list[i] = merged_event
            del m_detected_events_list[i + 1]
        else:
            i += 1
    if print_results:
        print("Temporal filter windows: {}".format(temporal_merging_window))
        print("Merged events by temporal filter: {}".format(merged_events))
        for dte in m_detected_events_list:
            color = colors[colors_key[classes.index(dte[2])]]
            plt.axvspan(dte[0], dte[1], 0, 0.33, color=color)
        plt.axhline(y=0.33, color='k', linestyle='-')
        plt.axhline(y=0.66, color='k', linestyle='-')
        # tolerance windows after a ground truth event
        print("Tolerance windows: {}".format(tolerance_window))

    flagged_events = list()

    for gte in ground_truth_events_list:
        start_time_gte = gte[0]
        end_time_gte = gte[1]
        label = gte[2]
        if label == 0:
            dtes = [dte for dte in detected_events_list if start_time_gte + tolerance_window <= dte[0] < end_time_gte]
            if len(dtes) == 0:
                flagged_events += [(gte[0], gte[1], gte[2], gte[2], 'TN', 'TN', int((gte[1] - gte[0]) / 1000))]
            else:
                flagged_events += [
                    (start_time_gte, dtes[0][0], label, label, 'TN', 'TN', int((dtes[0][0] - start_time_gte) / 1000))]
                for i in range(len(dtes) - 1):
                    dte = dtes[i]
                    flagged_events += [(dte[0], dte[1], dte[2], label, 'FP', 'FPI', 1)]
                    flagged_events += [
                        (dte[1], dtes[i + 1][0], label, label, 'TN', 'TN', int((dtes[i + 1][0] - dte[1]) / 1000))]
                flagged_events += [(dtes[-1][0], dtes[-1][1], dtes[-1][2], label, 'FP', 'FPI', 1)]
                flagged_events += [
                    (dtes[-1][1], end_time_gte, label, label, 'TN', 'TN', int((end_time_gte - dtes[-1][1]) / 1000))]
        else:
            # True positive and false positive merging
            tp_fpm = [dte for dte in detected_events_list if
                      start_time_gte <= dte[0] < end_time_gte + tolerance_window and dte[2] == label]
            fp_ds = [dte for dte in detected_events_list if
                     start_time_gte <= dte[0] < end_time_gte + tolerance_window and dte[2] != label]
            if len(tp_fpm) > 0:
                flagged_events += [(dte[0], dte[1], dte[2], label, 'FP', 'FPM', 1) for dte in tp_fpm]
                flagged_events[-1] = (tp_fpm[-1][0], tp_fpm[-1][1], tp_fpm[-1][2], label, 'TP', 'TP', 1)
                if len(fp_ds) > 0:
                    flagged_events += [(dte[0], dte[1], dte[2], label, 'FP', 'FPI', 1) for dte in fp_ds]
            elif len(fp_ds) > 0:
                flagged_events += [(dte[0], dte[1], dte[2], label, 'FP', 'FPI', 1) for dte in fp_ds]
                flagged_events[-1] = (fp_ds[-1][0], fp_ds[-1][1], fp_ds[-1][2], label, 'FP', 'FPS', 1)
            else:
                flagged_events += [(gte[0], gte[1], 0, label, 'FN', 'FND', 1)]
    flagged_events.sort(key=lambda tup: tup[0])

    total_events = functools.reduce(lambda x, y: x + y, [e[6] for e in flagged_events], 0)
    true_positive = functools.reduce(lambda x, y: x + y, [e[6] for e in flagged_events if e[4] == 'TP'], 0)
    true_negative = functools.reduce(lambda x, y: x + y, [e[6] for e in flagged_events if e[4] == 'TN'], 0)
    false_positive = functools.reduce(lambda x, y: x + y, [e[6] for e in flagged_events if e[4] == 'FP'], 0)
    false_negative = functools.reduce(lambda x, y: x + y, [e[6] for e in flagged_events if e[4] == 'FN'], 0)
    accuracy = (true_positive + true_negative) / total_events
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    if precision > 0 and recall > 0:
        f1 = 2 / (1 / recall + 1 / precision)
    else:
        f1 = 0
    if print_results:
        print("Accuracy: {}, Precision: {}, Recall: {}, F1: {}".format(accuracy, precision, recall, f1))

    tpr = true_positive / (true_positive + false_negative)
    fpr = false_positive / (true_negative + false_positive)
    if print_results:
        print("tpr: {}, fpr: {}".format(tpr, fpr))

    if compute_conf_matrix:
        cfm = compute_confusion_matrix_continuous(flagged_events, classes)
    else:
        cfm = None
    return [accuracy, precision, recall, f1, tpr, fpr, cfm]


def compute_confusion_matrix_continuous(flagged_events, classes, save_fig=False):
    num_classes = len(classes)
    # Main confusion matrix
    cf_matrix = np.zeros([num_classes + 1, num_classes + 1])
    # Additional matrix for FP-insertions
    fpi_matrix = np.zeros([num_classes, num_classes])
    # Additional array (diagonal of cf_matrix) for FP-merge
    fpm_array = np.zeros([num_classes])

    classes = np.append(classes, 0)

    for i in range(num_classes + 1):
        for j in range(num_classes + 1):
            # Diagonal
            if i == j:
                label = classes[i]
                # True negative. Null-class correctly recognized as null-class
                if label == 0:
                    tn = len([e for e in flagged_events if e[4] == 'TN'])
                    cf_matrix[i][j] = tn
                # True positive, FP-merging and FP-insertion for the diagonal (except NULL class -> True negative)
                else:
                    tp = functools.reduce(lambda x, y: x + y,
                                          [e[6] for e in flagged_events if e[4] == 'TP' and e[2] == label], 0)
                    cf_matrix[i][j] = tp
                    fpi = functools.reduce(lambda x, y: x + y,
                                           [e[6] for e in flagged_events if e[5] == 'FPI' and e[2] == label], 0)
                    fpi_matrix[i][j] = fpi
                    fpm = functools.reduce(lambda x, y: x + y,
                                           [e[6] for e in flagged_events if e[5] == 'FPM' and e[2] == label], 0)
                    fpm_array[i] = fpm
            # Rest of the matrix
            else:
                pred_label = classes[i]
                gt_label = classes[j]
                # Last column. Ground-truth null class
                if gt_label == 0:
                    fpi = functools.reduce(lambda x, y: x + y,
                                           [e[6] for e in flagged_events if
                                            e[5] == 'FPI' and e[2] == pred_label and e[3] == gt_label], 0)
                    cf_matrix[i][j] = fpi
                # Remaining matrix. First n-1 columns
                else:
                    fps = functools.reduce(lambda x, y: x + y,
                                           [e[6] for e in flagged_events if
                                            e[5] == 'FPS' and e[2] == pred_label and e[3] == gt_label], 0)
                    cf_matrix[i][j] = fps
                    if pred_label == 0:
                        fnd = functools.reduce(lambda x, y: x + y,
                                               [e[6] for e in flagged_events if
                                                e[5] == 'FND' and e[3] == gt_label], 0)
                        cf_matrix[i][j] = fnd

    print("Confusion matrix: ")
    print('\n'.join([''.join(['{:6}'.format(item) for item in row]) for row in cf_matrix]))

    # Normalize the confusion matrix
    # Normalize TP, FPs and FNd
    cf_matrix_normalized = cf_matrix
    cf_matrix_normalized[:, :-1] = cf_matrix[:, :-1] / np.sum(cf_matrix[:, :-1], axis=0)
    # Normalize FPi
    # total_null_event = len([e for e in flagged_events if e[4] == "TN"])
    # for i in range(len(flagged_events)-1):
    #     if flagged_events[i][3] == 0 and flagged_events[i]
    total_null_event = len([e for e in flagged_events if e[3] == 0])
    cf_matrix_normalized[:, -1] = cf_matrix[:, -1] / total_null_event

    print("Confusion matrix normalized: ")
    print('\n'.join([''.join(['{:6.2f}'.format(item) for item in row]) for row in cf_matrix_normalized]))

    print("FP-insertion matrix: ")
    print('\n'.join([''.join(['{:6.2f}'.format(item) for item in row]) for row in fpi_matrix]))

    print("FP-merge array: ")
    print(fpm_array)

    fig = plt.figure()
    subplt = fig.add_subplot(111)
    subplt.imshow(cf_matrix_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    fig.suptitle("Confusion matrix - Stream", y=1.05)
    tick_marks = np.arange(len(classes))
    subplt.set_xticks(tick_marks)
    subplt.set_xticklabels(classes)
    ax = plt.gca()
    ax.xaxis.tick_top()
    subplt.set_yticks(tick_marks)
    subplt.set_yticklabels(classes)
    subplt.set_xlabel('True label')
    subplt.set_ylabel('Predicted label')
    # plt.colorbar()
    thresh = cf_matrix.max() / 2.
    offset = 0.20
    for i, j in itertools.product(range(cf_matrix.shape[0]), range(cf_matrix.shape[1])):
        # True positive and False positive
        subplt.text(j, i, "{:04.2f}".format(cf_matrix[i, j]),
                    horizontalalignment="center",
                    color="white" if cf_matrix[i, j] > thresh else "black")
        if i == j and i != num_classes:
            subplt.text(j + offset, i - offset, "{:04.2f}".format(fpm_array[i]),
                        horizontalalignment="center",
                        color="white" if cf_matrix[i, j] > thresh else "blue")
    for i, j in itertools.product(range(fpi_matrix.shape[0]), range(fpi_matrix.shape[1])):
        subplt.text(j + offset, i + offset, "{:04.2f}".format(fpi_matrix[i, j]),
                    horizontalalignment="center",
                    color="yellow" if cf_matrix[i, j] > thresh else "red")
    if save_fig:
        plt.savefig(
            "/home/mathias/Documents/Academic/PhD/Publications/2018/ISWC/OptimizationWLCSS/figures/conf_matrix_continuous.eps",
            bbox_inches='tight', format='eps', dpi=1000)
    return cf_matrix


def event_extraction(prediction_array, classes):
    time = prediction_array[:, 0]
    test_events_dict = dict()
    ground_events_dict = dict()
    for c in classes:
        tmp_pred_labels = np.copy(prediction_array[:, 1])
        tmp_pred_labels[tmp_pred_labels != c] = 0

        diff_pred_labels = tmp_pred_labels[1:] - tmp_pred_labels[0:-1]
        diff_pred_labels = np.append(0, diff_pred_labels[0:-1])

        start_times_p = time[np.where(diff_pred_labels == c)]
        end_times_p = time[np.where(diff_pred_labels == -c)]

        test_events_dict[c] = [event for event in zip(start_times_p, end_times_p)]

        tmp_act_labels = np.copy(prediction_array[:, 2])
        tmp_act_labels[tmp_act_labels != c] = 0

        diff_act_labels = tmp_act_labels[1:] - tmp_act_labels[0:-1]
        diff_act_labels = np.append(0, diff_act_labels[0:-1])

        start_times_a = time[np.where(diff_act_labels == c)]
        end_times_a = time[np.where(diff_act_labels == -c)]

        ground_events_dict[c] = [event for event in zip(start_times_a, end_times_a)]

    tmp_act_labels = np.copy(prediction_array[:, 2])

    start_times_a = np.array(
        [time[i + 1] for i, e in enumerate(tmp_act_labels[:-1]) if e != 0 and tmp_act_labels[i + 1] == 0])
    end_times_a = np.array(
        [time[i] for i, e in enumerate(tmp_act_labels[:-1]) if e == 0 and tmp_act_labels[i + 1] != 0])
    start_times_a = np.append(time[0], start_times_a)
    end_times_a = np.append(end_times_a, time[-1])

    ground_events_dict[0] = [event for event in zip(start_times_a, end_times_a)]

    return test_events_dict, ground_events_dict

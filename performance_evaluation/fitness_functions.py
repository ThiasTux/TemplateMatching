import numpy as np


def isolated_fitness_function_params(matching_scores, thresholds, classes, parameter_to_optimize=5):
    num_classes = matching_scores.shape[1] - 1
    num_instances = matching_scores.shape[0]
    thresholds = np.append(thresholds, 0)
    scores = matching_scores - thresholds
    classes = np.unique(classes)
    true_positive = np.array(
        [np.count_nonzero(scores[np.where(scores[:, i] >= 0)[0]][:, -1] == classes[i]) for i in range(num_classes)])
    false_positive = np.array(
        [np.count_nonzero(scores[np.where(scores[:, i] >= 0)[0]][:, -1] != classes[i]) for i in range(num_classes)])
    true_negative = np.array(
        [np.count_nonzero(scores[np.where(scores[:, i] < 0)[0]][:, -1] != classes[i]) for i in range(num_classes)])
    false_negative = np.array(
        [np.count_nonzero(scores[np.where(scores[:, i] < 0)[0]][:, -1] == classes[i]) for i in range(num_classes)])
    if parameter_to_optimize == 1:
        # Accuracy
        return (np.sum(true_positive) + np.sum(true_negative)) / (num_classes * num_instances)
    elif parameter_to_optimize == 2:
        # Precision
        tps = np.sum(true_positive)
        fps = np.sum(false_positive)
        if tps != 0 or fps != 0:
            return tps / (tps + fps)
        else:
            return 0
    elif parameter_to_optimize == 3:
        # Recall
        tps = np.sum(true_positive)
        fns = np.sum(false_negative)
        if tps != 0 or fns != 0:
            return tps / (tps + fns)
        else:
            return 0
    elif parameter_to_optimize == 4:
        tps = np.sum(true_positive)
        fps = np.sum(false_positive)
        fns = np.sum(false_negative)
        # F1
        if tps != 0 or fps != 0:
            precision = tps / (tps + fps)
        else:
            precision = 0
        if tps != 0 or fns != 0:
            recall = tps / (tps + fns)
        else:
            recall = 0
        if precision != 0 and recall != 0:
            f1 = 2 / (1 / recall + 1 / precision)
            return f1
        else:
            return 0
    elif parameter_to_optimize == 5:
        tps = np.sum(true_positive)
        fps = np.sum(false_positive)
        fns = np.sum(false_negative)
        # F1
        if tps != 0 or fps != 0:
            precision = tps / (tps + fps)
        else:
            precision = 0
        if tps != 0 or fns != 0:
            recall = tps / (tps + fns)
        else:
            recall = 0
        if precision != 0 and recall != 0:
            f1 = 2 / (1 / recall + 1 / precision)
        else:
            f1 = 0
        accuracy = (np.sum(true_positive) + np.sum(true_negative)) / (num_classes * num_instances)
        return f1 * accuracy

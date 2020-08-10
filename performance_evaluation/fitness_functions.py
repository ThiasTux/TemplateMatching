import numpy as np
from scipy.special import expit


def isolated_fitness_function_params(matching_scores, labels, thresholds, classes, parameter_to_optimize='f1_acc'):
    num_classes = len(classes)
    num_instances = len(matching_scores)
    # scores = (matching_scores - thresholds) / thresholds
    # pred_labels = np.zeros(num_instances)
    # for i in range(num_instances):
    #     max_score = np.max(scores[i])
    #     if max_score > 0:
    #         pred_labels[i] = classes[np.argmax(scores[i])]
    true_positive = np.zeros([num_classes])
    false_positive = np.zeros([num_classes])
    true_negative = np.zeros([num_classes])
    false_negative = np.zeros([num_classes])
    for i in range(num_instances):
        for j in range(num_classes):
            act = classes[j]
            label = labels[i]
            test_matching_scores = matching_scores[i, j]
            if test_matching_scores >= thresholds[j] and act == label:
                true_positive[j] += 1
            elif test_matching_scores < thresholds[j] and act == label:
                false_negative[j] += 1
            elif test_matching_scores >= thresholds[j] and act != label:
                false_positive[j] += 1
            elif test_matching_scores < thresholds[j] and act != label:
                true_negative[j] += 1
    if parameter_to_optimize == 'acc':
        # Accuracy
        return (np.sum(true_positive) + np.sum(true_negative)) / (num_classes * num_instances)
    elif parameter_to_optimize == 'prec':
        # Precision
        tps = np.sum(true_positive)
        fps = np.sum(false_positive)
        if tps != 0 or fps != 0:
            return tps / (tps + fps)
        else:
            return 0
    elif parameter_to_optimize == 'recall':
        # Recall
        tps = np.sum(true_positive)
        fns = np.sum(false_negative)
        if tps != 0 or fns != 0:
            return tps / (tps + fns)
        else:
            return 0
    elif parameter_to_optimize == 'f1':
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
        return f1
    elif parameter_to_optimize == 'f1_acc':
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


def isolated_fitness_function_templates(scores, labels, threshold, parameter_to_optimize=1):
    good_scores = scores[labels != 0]
    bad_scores = scores[labels == 0]
    if len(good_scores) == 0 or len(bad_scores) == 0:
        return 0
    if parameter_to_optimize == 1:
        return np.mean(good_scores) - np.mean(bad_scores)
        # min(good) - max(bad)
    elif parameter_to_optimize == 2:
        min_good_scores = np.mean(good_scores[good_scores.argsort()[:10]])
        max_bad_scores = np.mean(bad_scores[bad_scores.argsort()[10:]])
        return np.array([min_good_scores - max_bad_scores, min_good_scores, max_bad_scores])
    # 90 % (good) - 10 % (bad)
    elif parameter_to_optimize == 3:
        good_percentile = np.percentile(good_scores, 10)
        bad_percentile = np.percentile(bad_scores, 90)
        return np.array([good_percentile - bad_percentile, good_percentile, bad_percentile])
    elif parameter_to_optimize == 4:
        avg_good = np.sum(good_scores - threshold) / len(good_scores)
        avg_bad = np.sum(threshold - bad_scores) / len(bad_scores)
        return avg_good + avg_bad
    elif parameter_to_optimize == 5:
        avg_good = np.sum(good_scores - threshold) / len(good_scores) ** 2
        avg_bad = (np.sum(bad_scores - threshold) / len(bad_scores)) ** 2
        return avg_good * (-avg_bad)
    elif parameter_to_optimize == 6:
        return (np.min(good_scores) - threshold) * (np.min(good_scores) - np.max(bad_scores))
    elif parameter_to_optimize == 7:
        avg_good = np.sum(good_scores - threshold) / len(good_scores)
        avg_bad = np.sum(threshold - bad_scores) / len(bad_scores)
        if avg_good < 0:
            return avg_good
        else:
            return -avg_bad
    elif parameter_to_optimize == 8:
        good_perc = np.percentile(good_scores, 10)
        bad_perc = np.percentile(bad_scores, 90)
        return good_perc - bad_perc
    elif parameter_to_optimize == 81:
        good_perc = np.percentile(good_scores, 10)
        bad_perc = np.percentile(bad_scores, 90)
        good_distance = good_perc - threshold
        bad_distance = bad_perc - threshold
        return good_distance - bad_distance
    elif parameter_to_optimize == 82:
        good_perc = np.percentile(good_scores, 10)
        bad_perc = np.percentile(bad_scores, 90)
        # good_distance = good_perc - threshold
        # bad_distance = bad_perc - threshold
        return (1 / (1 + np.e ** (good_perc * .005)) * 1 / (1 + np.e ** (-bad_perc * .005))) * 4 * threshold
    elif parameter_to_optimize == 83:
        good_perc = np.percentile(good_scores, 10)
        bad_perc = np.percentile(bad_scores, 90)
        good_distance = good_perc - threshold
        bad_distance = bad_perc - threshold
        if good_distance >= 1 >= bad_distance:
            return good_distance * (-bad_distance) / 1000
        else:
            a = 4000
            return (a / ((1 + np.e ** ((-good_distance - 1000) * .005)) * (
                    1 + np.e ** ((bad_distance - 1000) * .005)))) - a
    elif parameter_to_optimize == 84:
        min_good = np.min(good_scores)
        max_bad = np.max(bad_scores)
        good_distance = min_good - threshold
        bad_distance = max_bad - threshold
        if good_distance >= 1 >= bad_distance:
            return good_distance * (-bad_distance) / 1000
        else:
            a = 10000
            return (a / ((1 + np.e ** ((-good_distance - 1000) * .002)) * (
                    1 + np.e ** ((bad_distance - 1000) * .002)))) - a
    elif parameter_to_optimize == 85:
        min_good = np.min(good_scores)
        max_bad = np.max(bad_scores)
        good_distance = (min_good - threshold) / 1000
        bad_distance = (max_bad - threshold) / 1000
        if good_distance >= 0 >= bad_distance:
            fitness_score = (2 ** good_distance) * (2 ** (-bad_distance)) * 2000
        else:
            fitness_score = - (good_distance ** 2 + bad_distance ** 2) * 10000
        return fitness_score
    elif parameter_to_optimize == 86:
        good_perc = np.percentile(good_scores, 10)
        bad_perc = np.percentile(bad_scores, 90)
        # good_perc = np.min(good_scores)
        # bad_perc = np.max(bad_scores)
        good_distance = good_perc - threshold
        bad_distance = bad_perc - threshold
        if good_distance >= 0 and bad_distance >= 0:
            fitness_score = -(good_distance + bad_distance)
        elif good_distance >= 0 >= bad_distance:
            fitness_score = good_distance + (-bad_distance)
        elif good_distance <= 0 and bad_distance <= 0:
            fitness_score = good_distance + bad_distance
        elif good_distance <= 0 <= bad_distance:
            fitness_score = good_distance + (-bad_distance)
        return np.array([fitness_score, good_distance, bad_distance])
    elif parameter_to_optimize == 9:
        good_perc = np.percentile(good_scores, 10)
        bad_perc = np.percentile(bad_scores, 90)
        return (good_perc ** 2 - threshold ** 2) * (threshold ** 2 - bad_perc ** 2)
    elif parameter_to_optimize == 10:
        # Accuracy
        true_positive = len(good_scores[good_scores >= threshold])
        true_negative = len(bad_scores[bad_scores >= threshold])
        return (true_positive + true_negative) / len(scores)
    elif parameter_to_optimize == 11:
        good_distances = good_scores - threshold
        bad_distances = threshold - bad_scores
        good = np.sum(expit(good_distances)) / len(good_scores)
        bad = np.sum(expit(bad_distances)) / len(bad_scores)
        return good + bad
    return None


def isolated_fitness_function_all(scores, labels, threshold, parameter_to_optimize=1):
    good_scores = scores[labels != 0]
    bad_scores = scores[labels == 0]
    good_perc = np.average(good_scores)
    bad_perc = np.average(bad_scores)
    good_distance = good_perc - threshold
    bad_distance = bad_perc - threshold
    if good_distance >= 0 and bad_distance >= 0:
        fitness_score = -(good_distance + bad_distance)
    elif good_distance >= 0 >= bad_distance:
        fitness_score = good_distance + (-bad_distance)
    elif good_distance <= 0 and bad_distance <= 0:
        fitness_score = good_distance + bad_distance
    elif good_distance <= 0 <= bad_distance:
        fitness_score = good_distance + (-bad_distance)
    return fitness_score

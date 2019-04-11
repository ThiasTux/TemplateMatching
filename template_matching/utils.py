import numpy as np
from scipy.signal import find_peaks_cwt


def find_peaks(matching_scores):
    # Peaks found using scipy signal library
    peaks = [find_peaks_cwt(matching_scores[:, i], np.arange(70, 150)) for i in range(1, matching_scores.shape[1])]
    # Peaks found using Search Max for local maxima
    # peaks = wlcss.find_local_maxima(matching_scores[i], wsize=wsize)
    # if len(peaks) > 0:
    #     y = [matching_scores[i][x] for x in peaks if matching_scores[i][x] >= thresholds[i]]
    #     peaks = [x for x in peaks if matching_scores[i][x] >= thresholds[i]]
    #     subplt.plot(peaks, y, 'x', color='r', markersize=5)
    # peaks = [find_local_maxima(matching_scores[:, i]) for i in range(1, matching_scores.shape[1])]
    return peaks


def find_local_maxima(indata, wsize=1):
    size = len(indata)
    maxima_idxs = np.array([], dtype=int)
    flag = False
    first = True
    k = 0
    curmax = -float("inf")
    if wsize < 0:
        wsize = -wsize
        first = False
    for i in range(1, size):
        if flag:
            k += 1
        if not flag and indata[i] > indata[i - 1]:
            flag = True
            idx = 1
            k = 0
            curmax = indata[i]
        if first:
            if indata[i] > curmax:
                k = 0
                curmax = indata[i]
                idx = i
        else:
            if indata[i] > curmax:
                k = 0
                curmax = indata[i]
                idx = i
        if flag and k >= wsize and indata[i] < curmax:
            flag = False
            curmax = -float("inf")
            maxima_idxs = np.append(maxima_idxs, idx)
    return maxima_idxs

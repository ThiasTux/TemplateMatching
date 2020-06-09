"""
Plot Fitness Function for Template Generation
"""
import glob
import os

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


def fitness_function1(X, Y):
    rows = len(X)
    columns = len(Y)
    Z = np.zeros((rows, columns))
    for i in range(rows):
        for j in range(columns):
            good_distance = X[i, j]
            bad_distance = Y[i, j]
            if good_distance >= 1 >= bad_distance:
                Z[i, j] = good_distance * (-bad_distance) / 1000
            else:
                a = 10000
                Z[i, j] = (a / ((1 + np.e ** ((-good_distance - 1000) * .002)) * (
                        1 + np.e ** ((bad_distance - 1000) * .002)))) - a
    return Z


def fitness_function2(X, Y):
    rows = len(X)
    columns = len(Y)
    Z = np.zeros((rows, columns))
    for i in range(rows):
        for j in range(columns):
            good_distance = X[i, j]
            bad_distance = Y[i, j]
            if good_distance >= 0 >= bad_distance:
                Z[i, j] = (good_distance + (-bad_distance)) * 100000
            else:
                Z[i, j] = -(good_distance ** 2 + bad_distance ** 2)
    return Z


def fitness_function3(X, Y):
    rows = len(X)
    columns = len(Y)
    Z = np.zeros((rows, columns))
    for i in range(rows):
        for j in range(columns):
            good_distance = (X[i, j]) / 1000
            bad_distance = (Y[i, j]) / 1000
            if good_distance >= 0 >= bad_distance:
                # good_distance /= 1000
                # bad_distance /= 1000
                Z[i, j] = (2 ** good_distance) * (2 ** (-bad_distance)) * 2000
            else:
                Z[i, j] = - (good_distance ** 2 + bad_distance ** 2) * 10000
    return Z


def fitness_function4(X, Y):
    rows = len(X)
    columns = len(Y)
    Z = np.zeros((rows, columns))
    for i in range(rows):
        for j in range(columns):
            good_distance = (X[i, j]) / 1000
            bad_distance = (Y[i, j]) / 1000
            if good_distance >= 0 >= bad_distance:
                Z[i, j] = (1.5 ** good_distance) * (1.5 ** (-bad_distance)) * 100
            else:
                Z[i, j] = - (np.e ** (abs(good_distance) + abs(bad_distance))) * 10
    return Z


def fitness_function5(X, Y):
    rows = len(X)
    columns = len(Y)
    Z = np.zeros((rows, columns))
    for i in range(rows):
        for j in range(columns):
            good_distance = (X[i, j])
            bad_distance = (Y[i, j])
            if good_distance >= 0 and bad_distance >= 0:
                Z[i, j] = -(good_distance + bad_distance)
            elif good_distance >= 0 >= bad_distance:
                Z[i, j] = good_distance + (-bad_distance)
            elif good_distance <= 0 and bad_distance <= 0:
                Z[i, j] = good_distance + bad_distance
            elif good_distance <= 0 <= bad_distance:
                Z[i, j] = good_distance + (-bad_distance)
    return Z


if __name__ == '__main__':
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    X = np.arange(-25000, 25000, 200)
    Y = np.arange(-25000, 25000, 200)
    X, Y = np.meshgrid(X, Y)
    Z = (fitness_function5(X, Y))

    theCM = cm.coolwarm
    theCM._init()
    alphas = np.abs(np.linspace(-1.0, 1.0, theCM.N))
    theCM._lut[:-3, -1] = alphas
    ax.plot_surface(X, Y, Z, cmap=theCM, linewidth=0, antialiased=False, zorder=1)

    input_path = "/home/mathias/Documents/Academic/PhD/Research/WLCSSTraining/training/cuda/synthetic4/templates/zeus_templates_2020-02-03_16-21-20"

    # markers = ['o', '^', '*']
    conf_path = input_path + "_conf.txt"
    with open(conf_path, 'r') as conf_file:
        classes_line_num = 2
        for _ in range(classes_line_num):
            classes_line = conf_file.readline()
        classes = classes_line.split(":")[1].strip().split(" ")
    # classes = [407521]
    for j, c in enumerate(classes):
        scores_file = [file for file in glob.glob(input_path + "*_{}_scores.txt".format(c)) if
                       os.stat(file).st_size != 0][0]
        scores = np.loadtxt(scores_file, delimiter=",")
        good_distances = scores[:, -2]
        bad_distances = scores[:, -1]
        trace = np.zeros((len(good_distances)))
        t = np.arange(len(good_distances))
        for i in t:
            good_distance = good_distances[i]
            bad_distance = bad_distances[i]
            if good_distance >= 0 and bad_distance >= 0:
                trace[i] = -(good_distance + bad_distance)
            elif good_distance >= 0 >= bad_distance:
                trace[i] = good_distance + (-bad_distance)
            elif good_distance <= 0 and bad_distance <= 0:
                trace[i] = good_distance + bad_distance
            elif good_distance <= 0 <= bad_distance:
                trace[i] = good_distance + (-bad_distance)
        ax.scatter(good_distances, bad_distances, trace, c=t, zorder=20, label=c)

    ax.legend()

    ax.set_xlabel('Good distance')
    ax.set_ylabel('Bad distance')
    ax.set_zlabel('Fitness function')

    plt.show()

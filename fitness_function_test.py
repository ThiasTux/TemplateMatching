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
            if good_distance >= 1 >= bad_distance:
                Z[i, j] = good_distance * (-bad_distance)
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

    X = np.arange(-3000, 3000, 20)
    Y = np.arange(-3000, 3000, 20)
    X, Y = np.meshgrid(X, Y)
    Z = (fitness_function5(X, Y))

    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    ax.set_xlabel('Good distance')
    ax.set_ylabel('Bad distance')
    ax.set_zlabel('Fitness function')

    plt.show()

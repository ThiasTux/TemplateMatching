from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np


def fitness_function(X, Y, threshold):
    rows = len(X)
    columns = len(Y)
    Z = np.zeros((rows, columns))
    for i in range(rows):
        for j in range(columns):
            good_distance = X[i]
            bad_distance = Y[j]
            if good_distance >= 1 and bad_distance:
                # Z[i, j] = good_distance * bad_distance / 1000
                Z[i, j] = -10000
            else:
                a = 4000
                Z[i, j] = (a / ((1 + np.e ** ((-bad_distance - threshold) * .005)) * (
                            1 + np.e ** ((good_distance - threshold) * .005)))) - a
    return Z


if __name__ == '__main__':
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # # Make data.
    # # -/-
    # X1 = np.arange(-500, 110, 10)
    # Y1 = np.arange(-500, 110, 10)
    # X1, Y1 = np.meshgrid(X1, Y1)
    # Z1 = -1/(X1*Y1)
    #
    # # +/-
    # X2 = np.arange(100, 500, 10)
    # Y2 = np.arange(-500, 110, 10)
    # X2, Y2 = np.meshgrid(X2, Y2)
    # Z2 = X2 + Y2
    #
    # # -/+
    # X3 = np.arange(-500, 110, 10)
    # Y3 = np.arange(100, 500, 10)
    # X3, Y3 = np.meshgrid(X3, Y3)
    # Z3 = X3 + Y3
    #
    # # +/+
    # X4 = np.arange(100, 500, 10)
    # Y4 = np.arange(100, 500, 10)
    # X4, Y4 = np.meshgrid(X4, Y4)
    # Z4 = X4 * Y4

    # ax.plot_surface(X1, Y1, Z1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # ax.plot_surface(X2, Y2, Z2, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # ax.plot_surface(X3, Y3, Z3, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # ax.plot_surface(X4, Y4, Z4, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    X = np.arange(-2000, 2000, 10)
    Y = np.arange(-2000, 2000, 10)
    Z = (fitness_function(X, Y, 1000))
    X, Y = np.meshgrid(X, Y)


    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # ax.set_zscale('log')
    # ax.plot_surface(X4, Y4, Z4, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    ax.set_xlabel('Good distance')
    ax.set_ylabel('Bad distance')
    ax.set_zlabel('Fitness function')

    plt.show()

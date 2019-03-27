import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


def fitness_function(X, Y):
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
                a = 4000
                Z[i, j] = (a / ((1 + np.e ** ((-good_distance - 1000) * .005)) * (
                        1 + np.e ** ((bad_distance - 1000) * .005)))) - a
    return Z


if __name__ == '__main__':
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    X = np.arange(-2000, 2000, 10)
    Y = np.arange(-2000, 2000, 10)
    X, Y = np.meshgrid(X, Y)
    Z = (fitness_function(X, Y, 1000))

    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    ax.set_xlabel('Good distance')
    ax.set_ylabel('Bad distance')
    ax.set_zlabel('Fitness function')

    plt.show()

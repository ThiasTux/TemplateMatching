import math

import numpy as np

codebook_builder = {7: {  # 0: [0., 0., 0.],
    1: [-1., 0., 0.],
    2: [0., -1., 0.],
    3: [1., 0., 0.],
    4: [0., 1., 0.],
    5: [0., 0., 1.],
    6: [0., 0., -1]},
    15: {  # 0: [0., 0., 0.],
        1: [-1., 0., 0.],
        2: [0., -1., 0.],
        3: [1., 0., 0.],
        4: [0., 1., 0.],
        5: [0., 0., 1.],
        6: [0., 0., -1],
        7: [-1, 1, -1],
        8: [-1, -1, -1],
        9: [1, -1, -1],
        10: [1, 1, -1],
        11: [-1, 1, 1],
        12: [-1, -1, 1],
        13: [1, -1, 1],
        14: [1, 1, 1]},
    27: {  # 0: [0., 0., 0.],
        1: [-1., 0., 0.],
        2: [0., -1., 0.],
        3: [1., 0., 0.],
        4: [0., 1., 0.],
        5: [0., 0., 1.],
        6: [0., 0., -1],
        7: [-1, 1, -1],
        8: [-1, -1, -1],
        9: [1, -1, -1],
        10: [1, 1, -1],
        11: [-1, 1, 1],
        12: [-1, -1, 1],
        13: [1, -1, 1],
        14: [1, 1, 1],
        15: [-1, 0.0, -1],
        16: [0.0, -1, -1],
        17: [1, 0.0, -1],
        18: [0.0, 1, -1],
        19: [-1, -1, 0.0],
        20: [1, -1, 0.0],
        21: [1, 1, 0.0],
        22: [-1, 1, 0.0],
        23: [-1, 0.0, 1],
        24: [0.0, 1, 1],
        25: [1, 0.0, 1],
        26: [0.0, -1, 1]}}

keys = {7: [0, 1, 2, 3, 4, 5, 6],
        15: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        27: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]}


def create_3d_codebook(num_vectors):
    builder = codebook_builder[num_vectors]
    codebook = {k: builder[k] for k in builder.keys()}
    return codebook


def create_2d_codebook(num_vectors):
    if num_vectors == 4:
        v = [[1, 0], [0, 1], [-1, 0], [0, -1]]
    elif num_vectors == 8:
        v = [[1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]]
    elif num_vectors == 12:
        v = [[1, 0], [0.866, 0.5], [0.5, 0.866],
             [0, 1], [-0.5, 0.866], [-0.866, 0.5],
             [-1, 0], [-0.866, -0.5], [-0.5, -0.866],
             [0, -1], [0.5, -0.866], [0.866, -0.5]]
    else:
        return None
    codebook = {keys[15][i]: tuple(v[i]) for i in range(len(v))}
    return codebook


def normalize(v):
    norm = np.linalg.norm(v, ord=1)
    if norm == 0:
        return v
    return v / norm


def encode_2d(v, codebook=create_2d_codebook(8)):
    distances = {k - 1: angle_between(codebook.get(k), v) for k in codebook.keys()}
    return min(distances, key=distances.get)


def eucl_dist(v1, v2):
    dist = [(a - b) ** 2 for a, b in zip(v1, v2)]
    distance = math.sqrt(sum(dist))
    return distance


def encode_3d(v, alphabet=15):
    codebook = create_3d_codebook(alphabet)
    distances = {k - 1: angle_between(codebook.get(k), v) for k in codebook.keys()}
    return min(distances, key=distances.get)


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            # >>> angle_between((1, 0, 0), (0, 1, 0))
            # 1.5707963267948966
            # >>> angle_between((1, 0, 0), (1, 0, 0))
            # 0.0
            # >>> angle_between((1, 0, 0), (-1, 0, 0))
            # 3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def compute_3d_distance_matrix(print_output=True):
    codebook = create_3d_codebook(27)
    codebook_keys = sorted(list(codebook.keys()))
    matrix_distance = np.zeros((26, 26))
    for k_i in codebook_keys:
        v_1 = codebook[k_i]
        for k_j in codebook_keys:
            v_2 = codebook[k_j]
            matrix_distance[k_i - 1, k_j - 1] = angle_between(v_1, v_2)
    matrix_distance = (matrix_distance * 10).astype(int)
    if print_output:
        for i in range(26):
            print("{", end="")
            for j in range(25):
                print(matrix_distance[i, j], end=", ")
            print(matrix_distance[i, 25], end="},")
            print()


def compute_2d_distance_matrix(print_output=True):
    codebook = create_2d_codebook(8)
    codebook_keys = sorted(list(codebook.keys()))
    print(codebook_keys)
    matrix_distance = np.zeros((8, 8))
    for k_i in codebook_keys:
        for k_j in codebook_keys:
            matrix_distance[k_i - 1, k_j - 1] = angle_between(codebook[k_i], codebook[k_j])
    matrix_distance = (matrix_distance * 10).astype(int)
    if print_output:
        for i in range(8):
            print("{", end="")
            for j in range(7):
                print(matrix_distance[i, j], end=", ")
            print(matrix_distance[i, 7], end="},")
            print()


if __name__ == '__main__':
    compute_3d_distance_matrix()
    compute_2d_distance_matrix()

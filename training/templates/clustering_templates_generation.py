import numpy as np
from template_matching import wlcss_c as wlcss


class WLCSS_K_Means:
    """
    Custom K_Means using WLCSS as distance measure between each gesture and the cluster's centroid
    """

    def __init__(self, num_clusters=2, tol=0.001, max_iter=300, penalty=1, reward=8, acceptance_distance=3,
                 use_random_centroids=False):
        self.centroids = {}
        self.num_clusters = num_clusters
        self.tol = tol
        self.max_iter = max_iter
        self.penalty = penalty
        self.reward = reward
        self.acceptance_distance = acceptance_distance
        self.use_random_centroids = use_random_centroids

    def fit(self, data):
        """
        Training function to create clusters

        :param data: ndarray NxM containing N gesture, each with M samples
        """

        if self.use_random_centroids:
            np.random.seed(2)
            centroids_indexes = np.random.randint(0, len(data), size=self.num_clusters)
            # centroids_indexes = [1, 40]
            print(centroids_indexes)
            for i in range(self.num_clusters):
                self.centroids[i] = data[centroids_indexes[i]]
        else:
            for i in range(self.num_clusters):
                self.centroids[i] = data[i]

        for i in range(self.max_iter):
            self.classifications = {j: [] for j in range(self.num_clusters)}

            for featureset in data:
                distances = [wlcss.compute_wlcss(featureset, self.centroids[centroid], self.penalty, self.reward,
                                                 self.acceptance_distance)[0] for centroid in self.centroids]
                classification = distances.index(max(distances))
                self.classifications[classification].append(featureset)

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)

            optimized = True
            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                diff = abs(np.sum((current_centroid - original_centroid) / original_centroid) * 100.0)
                if diff > self.tol:
                    optimized = False

            if optimized:
                print("Iteration: {}".format(i))
                break

    def get_centroids(self):
        return self.centroids

    def predict(self, data):
        """
        Predict cluster the data would belong to using WLCSS

        :param data:ndarray
            single gesture to be classifier
        :return: classification
            index of the cluster closest to the gesture
        """
        distances = [wlcss.compute_wlcss(data, self.centroids[centroid], self.penalty, self.reward,
                                         self.acceptance_distance) for centroid in self.centroids]
        classification = distances.index(max(distances))
        return classification

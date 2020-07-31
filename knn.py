import scipy.spatial
import numpy as np
from collections import Counter

class KNN:

    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self.get_neighbors(x) for x in X]
        return np.array(y_pred)

    def get_neighbors(self, x):
        # Compute distances between x and all examples in the training set
        dist = [scipy.spatial.distance.euclidean(x, x_train) for x_train in self.X_train]
        # Sort by distance and return indices of the first k neighbors
        first_neighbors = np.argsort(dist)[:self.k]
        # Extract the labels of the k nearest neighbor training samples
        for i in first_neighbors:
            return self.y_train[i]
        k_neighbor_labels = self.y_train[i]
        # return the most common class label
        nearest_neighbors = Counter(k_neighbor_labels).most_common(1)
        return nearest_neighbors[0][0]
      
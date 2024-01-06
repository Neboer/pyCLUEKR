import numpy as np


class KNN:
    # y is plain vector, X is all training data.
    # X shape is n*m, m is the count of coordinate of each point, n is the count of all points.
    def __init__(self, X: np.ndarray, y: np.ndarray, k=3):
        assert X.shape[0] == y.shape[0]
        self.X_train = X
        self.y_train = y
        self.k = k

    def predict(self, X):
        distances = np.linalg.norm(self.X_train - X, axis=1)
        nearest_neighbor_ids = distances.argsort()[:self.k]
        # print(nearest_neighbor_ids)
        nearest_neighbor_ys = self.y_train[nearest_neighbor_ids]
        prediction = nearest_neighbor_ys.mean()
        return prediction

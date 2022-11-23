from typing import Any

import numpy as np
from numpy import ndarray
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


def get_minkowski_distance(x1, x2, p):
    dist_sum = np.abs(np.sum((x1 - x2) ** p))
    return np.power(dist_sum, 1 / p)


class KNN(BaseEstimator):
    def __init__(self, k=3, p_minkowski=2):
        self.y = None
        self.x = None
        self.n_features_in_ = None
        self.k = k
        self.p_minkowski = p_minkowski

    def fit(self, x, y):
        x, y = check_X_y(x, y)

        self.n_features_in_ = x.shape[1]

        self.x = x
        self.y = y

        return self

    def predict(self, x) -> ndarray:
        check_is_fitted(self)
        x = check_array(x)
        y_prediction = [self._get_single_prediction(x_test_row) for x_test_row in x]

        return np.array(y_prediction)

    def _get_single_prediction(self, x_test_row) -> Any:
        distances = [
            get_minkowski_distance(x_test_row, x_train_row, self.p_minkowski)
            for x_train_row in self.x
        ]
        k_index = np.argsort(distances)[: self.k]
        k_labels = [self.y[idx] for idx in k_index]
        return np.argmax(np.bincount(k_labels))

    def score(self, x_test, y_test) -> float:
        predict = self.predict(x_test)
        return accuracy_score(y_test, predict)

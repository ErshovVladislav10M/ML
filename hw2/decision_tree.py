from typing import Any

import numpy as np
from numpy import ndarray
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class Node:
    def __init__(
        self, feature=None, threshold=None, left=None, right=None, *, value=None
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self) -> bool:
        return self.value is not None


def entropy(y) -> Any:
    proportions = np.bincount(y) / len(y)
    return -np.sum([p * np.log2(p) for p in proportions if p > 0])


def create_split(x, thresh):
    left_idx = np.argwhere(x <= thresh).flatten()
    right_idx = np.argwhere(x > thresh).flatten()
    return left_idx, right_idx


def information_gain(x, y, thresh):
    parent_loss = entropy(y)
    left_idx, right_idx = create_split(x, thresh)
    n, n_left, n_right = len(y), len(left_idx), len(right_idx)

    if n_left == 0 or n_right == 0:
        return 0

    child_loss = (n_left / n) * entropy(y[left_idx]) + (n_right / n) * entropy(
        y[right_idx]
    )
    return parent_loss - child_loss


def best_split(x, y, features):
    split = {"score": -1, "feat": None, "thresh": None}

    for feat in features:
        x_feat = x[:, feat]
        thresholds = np.unique(x_feat)
        for thresh in thresholds:
            score = information_gain(x_feat, y, thresh)

            if score > split["score"]:
                split["score"] = score
                split["feat"] = feat
                split["thresh"] = thresh

    return split["feat"], split["thresh"]


class DTC(BaseEstimator):
    def __init__(self, max_depth=100, min_samples_split=2):
        self.n_class_labels = None
        self.n_samples = None
        self.n_features = None
        self.root = None
        self.n_features_in_ = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def _is_finished(self, depth) -> bool:
        return (
            depth >= self.max_depth
            or self.n_class_labels == 1
            or self.n_samples < self.min_samples_split
        )

    def build_tree(self, x, y, depth=0) -> Node:
        self.n_samples, self.n_features = x.shape
        self.n_class_labels = len(np.unique(y))

        if self._is_finished(depth):
            most_common_label = np.argmax(np.bincount(y))
            return Node(value=most_common_label)

        best_feat, best_thresh = best_split(x, y, list(range(self.n_features)))

        left_idx, right_idx = create_split(x[:, best_feat], best_thresh)
        left_child = self.build_tree(x[left_idx, :], y[left_idx], depth + 1)
        right_child = self.build_tree(x[right_idx, :], y[right_idx], depth + 1)
        return Node(best_feat, best_thresh, left_child, right_child)

    def traverse_tree(self, x, node) -> Any:
        if node.is_leaf():
            return node.value

        if x[node.feature] <= node.threshold:
            return self.traverse_tree(x, node.left)
        return self.traverse_tree(x, node.right)

    def fit(self, x, y):
        x, y = check_X_y(x, y)
        self.n_features_in_ = x.shape[1]
        self.root = self.build_tree(x, np.array(y, dtype="int64"))
        return self

    def predict(self, x) -> ndarray:
        check_is_fitted(self)
        x = check_array(x)
        predictions = [self.traverse_tree(x, self.root) for x in x]
        return np.array(predictions)

    def score(self, x_test, y_test) -> float:
        predict = self.predict(x_test)
        return accuracy_score(y_test, predict)

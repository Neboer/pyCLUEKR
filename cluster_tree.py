# this method is from the paper DIRECTLY.
from __future__ import annotations
import numpy as np


# point_ids is one-dimensional array.
class HCTreeNode:
    def __init__(self, point_ids: np.ndarray, mean_X: np.ndarray, children=None):
        if children is None:
            children = []
        self.children: list[HCTreeNode] = children
        self.point_ids: np.ndarray = point_ids
        self.mean_X: np.ndarray = mean_X

    def __iter__(self):
        return iter(self.children)

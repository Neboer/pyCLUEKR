import numpy as np

from hierarchical_cluster import HICluster
from cluster_tree import HCTreeNode
from itertools import combinations


def test_far(d_i: float, d_j: float) -> bool:
    return min(d_i / d_j, d_j / d_i) < 0.9


def confidence(dist: list):
    far_l = [test_far(*c) for c in combinations(dist, 2)]
    return all(far_l)



def CLUEKR(query_X: np.ndarray, current_node: HCTreeNode, k: int):
    if len(current_node.children) >0:
        dist = []
        for node in current_node.children:
            # 找到与孩子节点之间的距离
            dist.append(np.linalg.norm(query_X - node.mean_X))
        if confidence(dist):
            nearest_node_index = np.argmin(dist)[0]
            nearest_node = current_node.children[nearest_node_index]
            return CLUEKR(query_X, nearest_node, k)
        # 如果不够自信可以使用孩子节点，或者压根就没有孩子节点，我们不得不对当前大节点进行KNN。
    return KNN(query_X, current_node, k)
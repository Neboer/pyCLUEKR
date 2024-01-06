# fill treenode's children with new trees.
import numpy as np

from cluster_tree import HCTreeNode


class HICluster:
    def __init__(self, X: np.ndarray, y: np.ndarray, max_depth=3):
        self.X_train = X
        self.y_train = y
        self.max_depth = max_depth
        self.root_node: HCTreeNode = None

    def _mean_X_of_cluster(self, cluster: list[int] | np.ndarray) -> np.ndarray:
        all_X_of_cluster = self.X_train[cluster]
        return np.mean(all_X_of_cluster, axis=0)

    # fill all nodes in tree.
    def _fill_child(self, tree_node: HCTreeNode):
        n = tree_node.point_ids.size
        # point id 是点在数据集中的位置。
        sorted_indices = self.y_train[tree_node.point_ids].argsort()
        sorted_point_ids = tree_node.point_ids[sorted_indices]
        # 根据y值排好序的 point_id

        # 找到n/4点和3n/4点。
        lower_point_id = sorted_point_ids[int(n / 4)]
        higher_point_id = sorted_point_ids[int(3 * n / 4)]

        lower_point_X = self.X_train[lower_point_id]
        higher_point_X = self.X_train[higher_point_id]
        center_point_X = (lower_point_X + higher_point_X) / 2

        cluster_1 = []
        cluster_2 = []
        cluster_c = []

        for point_id in tree_node.point_ids:
            current_point_X = self.X_train[point_id]
            distance_to_lower = np.linalg.norm(current_point_X - lower_point_X)
            distance_to_higher = np.linalg.norm(current_point_X - higher_point_X)
            distance_to_center = np.linalg.norm(current_point_X - center_point_X)

            if distance_to_lower <= distance_to_higher:
                cluster_1.append(point_id)
            else:
                cluster_2.append(point_id)

            if 0.9 < distance_to_lower / distance_to_center < 1.0 or 0.9 < distance_to_higher / distance_to_center < 1.0:
                cluster_c.append(point_id)

        tree_node.children.append(HCTreeNode(np.array(cluster_1), self._mean_X_of_cluster(cluster_1)))
        tree_node.children.append(HCTreeNode(np.array(cluster_2), self._mean_X_of_cluster(cluster_2)))
        tree_node.children.append(HCTreeNode(np.array(cluster_c), self._mean_X_of_cluster(cluster_c)))

    def _build_tree(self, tree_node: HCTreeNode, current_depth: int):
        if current_depth > self.max_depth:
            return None
        else:
            self._fill_child(tree_node)
            for node in tree_node:
                self._build_tree(node, current_depth + 1)

    def build_tree(self):
        self.root_node = HCTreeNode(np.arange(self.y_train.size), self._mean_X_of_cluster(np.arange(self.y_train.size)))
        self._build_tree(self.root_node, 1)

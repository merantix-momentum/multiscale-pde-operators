import itertools
from collections import defaultdict
from copy import deepcopy

import torch
from torch_geometric.utils import to_undirected

from multiscale_operator.transforms.cache import CachedTransform


class Leaf:
    def __init__(self, points, point_indices, parent=None):
        self.points = points
        self.point_indices = point_indices
        self.parent = parent

    def fill_positions(self, positions):
        positions[self.point_indices] = self.points
        positions[self.node_index] = self.pos

    def update_pos(self):
        self.pos = torch.mean(self.points, axis=0)

    def max_idx(self):
        return max(self.node_index, self.point_indices.max())

    def traverse(self):
        return [self]

    def range(self, i):
        return torch.min(self.points[:, i]), torch.max(self.points[:, i])

    def __repr__(self):
        return f"Leaf({self.points.shape[0]})"


class Node:
    def __init__(self, parent, node_index):
        self.parent = parent
        self.node_index = node_index

    def update_pos(self):
        self.left.update_pos()
        self.right.update_pos()
        self.pos = (self.left.pos + self.right.pos) / 2.0

    def max_idx(self):
        return max(self.left.max_idx(), self.right.max_idx(), self.node_index)

    def fill_positions(self, positions):
        positions[self.node_index] = self.pos
        self.left.fill_positions(positions)
        self.right.fill_positions(positions)

    def range(self, i):
        leftrange = self.left.range(i)
        rightrange = self.right.range(i)
        return min(leftrange[0], rightrange[0]), max(leftrange[1], rightrange[1])

    @staticmethod
    def from_leaves(parent, node_index, leftleaf, rightleaf):
        result = Node(parent, node_index)
        result.left = leftleaf
        result.right = rightleaf
        leftleaf.parent = result
        rightleaf.parent = result

        return result

    def remove_child(self, child):
        if self.left == child:
            self.left = None
        elif self.right == child:
            self.right = None
        else:
            raise ValueError("Leaf not found")

    def change_child(self, old, new):
        if self.left == old:
            self.left = new
        elif self.right == old:
            self.right = new
        else:
            raise ValueError("Leaf not found")

    def empty(self):
        return self.left is None and self.right is None

    def traverse(self, parent=None):
        return itertools.chain(self.left.traverse(), self.right.traverse())

    def __repr__(self):
        return f"Node({self.left}, {self.right})"


class TreeOpTransform(CachedTransform):
    def __init__(self, n_levels, k_hop_levels, k_neighbors):
        self.n_levels = n_levels
        self.k_hop_levels = k_hop_levels
        self.k_neighbors = k_neighbors
        super().__init__(check_attributes=["pos"])

    def split(self, leaf, i):
        median = torch.median(leaf.points[:, i])
        left = leaf.points[leaf.points[:, i] < median]
        right = leaf.points[leaf.points[:, i] >= median]

        dist = torch.abs(leaf.points - torch.mean(leaf.points, dim=0))
        dist_sq_sum = torch.sum(dist**2, axis=1)
        split_point_index = leaf.point_indices[torch.argmin(dist_sq_sum)]

        return (
            Leaf(left, leaf.point_indices[leaf.points[:, i] < median]),
            Leaf(right, leaf.point_indices[leaf.points[:, i] >= median]),
            median,
            split_point_index,
        )

    def create_tree(self, levels, points):
        tree = Leaf(points, torch.arange(points.shape[0]))
        idx = points.shape[0] - 1

        def idx_counter():
            nonlocal idx
            idx += 1
            return idx

        for _ in range(levels):
            for leaf in list(tree.traverse()):
                leftleaf, rightleaf, median, split_point_index, i = self.split_var(leaf)
                next_node = Node.from_leaves(leaf.parent, idx_counter(), leftleaf, rightleaf)
                next_node.separator = median
                next_node.split_point_index = split_point_index
                next_node.split_axis = i

                if leaf.parent is None:
                    tree = next_node  # this is our root node
                else:
                    leaf.parent.change_child(leaf, next_node)

        # fill in last indices
        for leaf in list(tree.traverse()):
            leaf.node_index = idx_counter()

        return tree

    def create_edge_indices_local(self, curr_nodes, k_hop_levels, idx_str="node_index"):
        result = []
        parent_nodes = []

        for node in curr_nodes:
            if node.parent is None:
                # skip root node
                continue

            parent_node = node
            for _ in range(k_hop_levels):
                # go up the tree
                if parent_node.parent is not None:
                    parent_node = parent_node.parent

            idx_node = getattr(node, idx_str)
            idx_parent_node = getattr(parent_node, idx_str)

            edge_index = torch.tensor([idx_parent_node, idx_node]).reshape(2, 1)
            result.append(edge_index)

            if parent_node not in parent_nodes:
                parent_nodes.append(parent_node)

        return result, parent_nodes

    def create_edge_indices_simple(self, tree, k_hop_levels=1):
        # simply use the median node to aggregate information across levels
        # do not add any nodes to the graph
        tree = deepcopy(tree)
        result = defaultdict(list)
        level = 1
        curr_nodes = []

        for leaf in list(tree.traverse()):
            if len(leaf.points) > 0 and leaf.parent is not None:
                # generate edge index connecting all leaf nodes
                # base connections

                edge_index = torch.stack(
                    [torch.full(leaf.point_indices.shape[0:1], leaf.parent.split_point_index), leaf.point_indices]
                )
                result[level].append(edge_index)
            curr_nodes.append(leaf.parent)

        while len(curr_nodes) > 1:
            level += 1
            edge_indices, curr_nodes = self.create_edge_indices_local(
                curr_nodes, k_hop_levels, idx_str="split_point_index"
            )

            if len(edge_indices) > 0:
                result[level] = edge_indices

        for lvl in result:
            result[lvl] = torch.hstack(result[lvl])
        return result

    def create_edge_indices(self, tree, k_hop_levels=1):
        tree = deepcopy(tree)
        result = defaultdict(list)
        level = 1
        curr_nodes = []

        for leaf in list(tree.traverse()):
            if len(leaf.points) > 0:
                # generate edge index connecting all leaf nodes
                # base connections
                edge_index = torch.stack(
                    [torch.full(leaf.point_indices.shape[0:1], leaf.node_index), leaf.point_indices]
                )
                result[level].append(edge_index)
            curr_nodes.append(leaf)

        while len(curr_nodes) > 1:
            level += 1
            edge_indices, curr_nodes = self.create_edge_indices_local(curr_nodes, k_hop_levels)

            if len(edge_indices) > 0:
                result[level] = edge_indices

        for lvl in result:
            result[lvl] = torch.hstack(result[lvl])
        return result

    def split_var(self, leaf):
        var = torch.var(leaf.points, axis=0)
        i = torch.argmax(var)
        return *self.split(leaf, i), i

    def postprocess(self, sample_in, sample_out):
        # aggreate edge indices
        tree_edge_indices = sample_out["tree_edge_indices"]
        # tree_level_indices = sample_out["tree_level_indices"]
        # , *tree_level_indices.values()
        sample_in.edge_index = torch.cat([sample_in.edge_index, *tree_edge_indices.values()], dim=1)
        return sample_in

    def transform(self, sample, ret_tree=False):
        tree = self.create_tree(self.n_levels, sample.pos)
        edge_indices = self.create_edge_indices_simple(tree, self.k_hop_levels)

        # to undirected
        for lvl in edge_indices:
            edge_indices[lvl] = to_undirected(edge_indices[lvl])

        return {
            "tree_edge_indices": edge_indices,
            "tree": tree if ret_tree else None,
        }

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import networkx as nx  # type: ignore

from ..factor_graph import FactorGraph
from ..strategy import Region, Strategy


@dataclass
class FowardBackwardStrategy(Strategy):
    group_schedule: List[Tuple[int, List[int]]] | None = None

    def edge_groups(self) -> List[Tuple[int, List[int]]]:
        return self.group_schedule if self.group_schedule is not None else super().edge_groups()


def BetheGraph(graph: FactorGraph, passes: int = 1) -> Strategy:
    r"""
    Returns the bethe graph region graph: A region for every factor and a
        separate region for each variable. Factor regions (including variables
        that they touch) point to each variable region for any variable touched
        by the factor. Inference with this strategy is equivalent to vanilla
        belief propagation.
    """
    # 1) pick an ordering of the nodes using dfs;
    # 2) for each factor in that order, pass messages from the factor to future nodes
    # 3) same thing in reverse order

    nx_graph = nx.from_dict_of_lists(dict(enumerate(graph.neighbors)))
    edge_groups = []

    def add_edges(node_order):
        visited = set()
        for node in node_order:
            visited.add(node)
            if graph.is_factor_node(node):
                future_neighbors = [neighbor for neighbor in graph.neighbors[node]
                                    if neighbor not in visited]
                if future_neighbors:
                    edge_groups.append((node, future_neighbors))

    order = list(nx.dfs_preorder_nodes(nx_graph))
    add_edges(order)
    add_edges(reversed(order))
    # TODO: add in blank factors for queries if necessary
    return FowardBackwardStrategy(
        regions=[
            Region(graph, (factor_node, *graph.neighbors[factor_node]), 1.0)
            for factor_node in graph.factor_nodes] + [
            Region(graph, (variable_node,), 1 - len(graph.neighbors[variable_node]))
            for variable_node in graph.variable_nodes],
        edges=[(i, j) for i in range(len(graph.factor_nodes)) for j in graph.neighbors[i]],
        passes=passes,
        group_schedule=edge_groups)

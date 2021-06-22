from __future__ import annotations

from ..factor_graph import FactorGraph
from ..strategy import Region, Strategy


def BetheGraph(graph: FactorGraph) -> Strategy:
    # TODO: add in blank factors for queries if necessary
    return Strategy(
        regions=[
            Region(graph, (factor_node, *graph.neighbors[factor_node]), 1.0)
            for factor_node in graph.factor_nodes] + [
            Region(graph, (variable_node,), 1 - len(graph.neighbors[variable_node]))
            for variable_node in graph.variable_nodes],
        edges=[(i, j) for i in range(len(graph.factor_nodes)) for j in graph.neighbors[i]])

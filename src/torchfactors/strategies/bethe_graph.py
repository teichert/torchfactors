from __future__ import annotations




def BetheTree(graph: FactorGraph) -> Strategy:
    # TODO: add in blank factors for queries if necessary
    return Strategy(
        regions=[
            Region((factor_node, *graph.neighbors[factor_node]), 1.0)
            for factor_node in graph.factor_nodes] + [
            Region((variable_node,), 1 - len(graph.neighbors[variable_node]))
            for variable_node in graph.variable_nodes],
        edges=[(i, j) for i in range(graph.num_nodes) for j in graph.neighbors[i]])

from abc import ABC, abstractmethod
from graph import Graph, HiddenInput, topological_sort
from warnings import warn
from typing import Any


class TransformationPass(ABC):
    @abstractmethod
    def apply(self, graph: Graph) -> Graph:
        pass


# TODO: test that the degree of the vertex does not change, i.e. that
# in_degree() + len(collapsed_inputs) = const
class ConstantHidingPass(TransformationPass):
    def apply(self, graph: Graph) -> Graph:
        to_remove: list[int] = []
        for node_id, node in graph.nodes.items():
            if node.node_type == "Constant":
                if graph.in_degree(node_id) > 0:
                    warn(
                        f"Constant {node_id} has input ports. Skipping this node")
                shared = graph.out_degree(node_id) > 1
                for edge_id in graph.outcoming_edges(node_id):
                    edge = graph.edges[edge_id]
                    to = graph.node(edge.to_node)
                    to.meta.hidden_inputs[edge.to_port] = HiddenInput(
                        edge.shape, edge.dtype, shared)
                to_remove.append(node_id)

        for node_id in to_remove:
            graph.remove_node(node_id)

        return graph


class HighlightingPass(TransformationPass):
    def __init__(self, names: list[str]) -> None:
        self.names = names

    def apply(self, graph: Graph) -> Graph:
        for node_id, node in graph.nodes.items():
            if node.node_type in self.names:
                graph.node(node_id).meta.border_color = 'red'

        return graph


class TruncationPass(TransformationPass):
    def __init__(self, size: int) -> None:
        self.size = size

    def apply(self, graph: Graph) -> Graph:
        top_order = topological_sort(graph)
        to_remove = top_order[self.size:]
        for node_id in to_remove:
            graph.remove_node(node_id)

        return graph
from abc import ABC, abstractmethod
from graph import Graph, HiddenInput, topological_sort
from warnings import warn

import time


class TransformationPass(ABC):
    @abstractmethod
    def apply(self, graph: Graph) -> Graph:
        pass


class Pipeline:
    def __init__(self, named_steps: list[tuple[str, TransformationPass]]):
        names = [name for name, _ in named_steps]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate step names in pipeline")

        self._named_steps = {step[0]: step[1] for step in named_steps}
        self._disabled = {step[0]: False for step in named_steps}
        self.timings: dict[str, float] = {}

    def _check_step(self, name: str):
        if not (name in self._named_steps):
            raise KeyError(f"Step {name} not found")

    def enable(self, name: str):
        self._check_step(name)
        self._disabled[name] = False

    def disable(self, name: str):
        self._check_step(name)
        self._disabled[name] = True

    def apply(self, graph: Graph) -> Graph:
        self.timings: dict[str, float] = {}
        for name, step in self._named_steps.items():
            if not self._disabled[name]:
                start = time.time()
                graph = step.apply(graph)
                self.timings[name] = time.time() - start
        return graph


class ConstantHidingPass(TransformationPass):
    def apply(self, graph: Graph) -> Graph:
        to_remove: list[int] = []
        for node_id, node in graph.nodes.items():
            if node.node_type == "Constant":
                if graph.in_degree(node_id) > 0:
                    warn(
                        f"Constant {node_id} has input ports. Skipping this node")
                    continue
                shared = graph.out_degree(node_id) > 1
                for edge_id in graph.outcoming_edges(node_id):
                    edge = graph.edges[edge_id]
                    to = graph.node(edge.to_node)
                    to.meta.hidden_inputs[edge.to_port] = HiddenInput(
                        edge.shape, edge.dtype, shared, node_id)
                to_remove.append(node_id)

        for node_id in to_remove:
            graph.remove_node(node_id)

        return graph


class HighlightingPass(TransformationPass):
    def __init__(self, names: set[str]) -> None:
        self.names = names

    def apply(self, graph: Graph) -> Graph:
        for node in graph.nodes.values():
            if node.name in self.names:
                node.meta.border_color = 'red'

        return graph


class TruncationPass(TransformationPass):
    def __init__(self, size: int = -1, offset: int = 0) -> None:
        if size < -1 or offset < 0:
            raise ValueError("size must be >= -1 and offset >= 0")
        self.size = size
        self.offset = offset

    def apply(self, graph: Graph) -> Graph:
        top_order = topological_sort(graph)
        # keep the window [offset, offset + size) of the topological order
        if self.size != -1:
            to_remove = top_order[:self.offset] + top_order[self.offset + self.size:]
        else:
            to_remove = top_order[:self.offset]
        for node_id in to_remove:
            graph.remove_node(node_id)

        return graph


class ProxyPass(TransformationPass):
    """
    Localizes each connection by making separate copies of shared nodes, that have more than max_consumers successors.
    Improves readability by eliminating the need to trace back a long edge.
    """

    def __init__(self, max_consumers=2) -> None:
        self.max_consumers = max_consumers
        pass

    def apply(self, graph: Graph) -> Graph:
        for node_id, node in graph.nodes.items():
            succ = {graph.edges[edge_id].to_node for edge_id in graph.outcoming_edges(node_id)}

            if len(succ) > self.max_consumers:
                node.meta.proxy_output = True

        return graph

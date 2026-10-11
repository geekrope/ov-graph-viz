import itertools

from collections import deque
from dataclasses import dataclass, field


@dataclass
class Edge:
    from_node: int
    from_port: int
    to_node: int
    to_port: int
    shape: tuple[str, ...] | None = None
    dtype: str | None = None


@dataclass
class HiddenInput:
    shape: tuple[str, ...] | None = None
    dtype: str | None = None
    shared: bool = False


@dataclass
class NodeMetadata:
    border_color: str | None = None
    hidden_inputs: dict[int, HiddenInput] = field(default_factory=dict)


@dataclass
class Node:
    name: str
    node_type: str
    attributes: dict[str, str] = field(default_factory=dict)
    meta: NodeMetadata = field(default_factory=NodeMetadata)


class Graph:
    def __init__(self) -> None:
        self.nodes: dict[int, Node] = {}
        self.edges: dict[int, Edge] = {}
        self._in: dict[int, set[int]] = {}
        self._out: dict[int, set[int]] = {}
        self._node_ids = itertools.count()
        self._edge_ids = itertools.count()

    def _check_node(self, node_id: int) -> None:
        if node_id not in self.nodes:
            raise KeyError(f"Node {node_id} not found")

    def add_node(self, node: Node) -> int:
        node_id = next(self._node_ids)

        self.nodes[node_id] = node
        self._in[node_id] = set()
        self._out[node_id] = set()

        return node_id

    def add_edge(self, edge: Edge) -> int:
        self._check_node(edge.from_node)
        self._check_node(edge.to_node)

        edge_id = next(self._edge_ids)

        self.edges[edge_id] = edge
        self._out[edge.from_node].add(edge_id)
        self._in[edge.to_node].add(edge_id)

        return edge_id

    def remove_edge(self, edge_id: int) -> None:
        edge = self.edges.pop(edge_id)

        self._out[edge.from_node].discard(edge_id)
        self._in[edge.to_node].discard(edge_id)

    def remove_node(self, node_id: int) -> None:
        self._check_node(node_id)

        for edge_id in list(self._in[node_id].union(self._out[node_id])):
            self.remove_edge(edge_id)

        del self.nodes[node_id], self._in[node_id], self._out[node_id]

    def node(self, node_id: int):
        self._check_node(node_id)
        return self.nodes[node_id]

    def incoming_edges(self, node_id: int):
        self._check_node(node_id)
        return self._in[node_id]

    def outcoming_edges(self, node_id: int):
        self._check_node(node_id)
        return self._out[node_id]

    def in_degree(self, node_id: int) -> int:
        self._check_node(node_id)
        return len(self._in[node_id])

    def out_degree(self, node_id: int) -> int:
        self._check_node(node_id)
        return len(self._out[node_id])

# computes the in degree of each node. when the node is processed decreses the
# degree of its children, which ensures that by the moment the degree reaches 0
# all the nodes connected to the current one were processed
def topological_sort(graph: Graph) -> list[int]:
    order: list[int] = []
    to_process: deque[int] = deque()
    in_degrees: dict[int, int] = {}

    for node_id in graph.nodes.keys():
        in_degrees[node_id] = graph.in_degree(node_id)
        if in_degrees[node_id] == 0:
            to_process.append(node_id)

    while len(to_process) > 0:
        top = to_process.popleft()
        order.append(top)
        for edge_id in graph.outcoming_edges(top):
            edge = graph.edges[edge_id]
            in_degrees[edge.to_node] -= 1
            if in_degrees[edge.to_node] == 0:
                to_process.append(edge.to_node)

    if len(order) != len(graph.nodes):
        raise ValueError(
            f"Graph contains a cycle: sorted {len(order)} of {len(graph.nodes)} nodes")

    return order

import itertools
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
class Node:
    name: str
    node_type: str
    attributes: dict[str, str] = field(default_factory=dict)


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

    def in_degree(self, node_id: int) -> int:
        self._check_node(node_id)
        return len(self._in[node_id])

    def out_degree(self, node_id: int) -> int:
        self._check_node(node_id)
        return len(self._out[node_id])
from graph import Graph, Node, Edge, topological_sort
from conftest import MODEL

import pytest

@pytest.fixture
def cyclic_graph():
    graph = Graph()
    A = graph.add_node(Node("A", "Add"))
    B = graph.add_node(Node("B", "Add"))
    C = graph.add_node(Node("C", "Add"))

    graph.add_edge(Edge(A, 0, B, 0))
    graph.add_edge(Edge(B, 0, C, 0))
    graph.add_edge(Edge(C, 0, A, 0))

    return graph


@pytest.mark.skipif(not MODEL.exists(), reason="TinyLlama not downloaded")
def test_topological_sort_inversions(tiny_llama_graph):
    order = topological_sort(tiny_llama_graph)
    positions = {node_id: i for i, node_id in enumerate(order)}

    for edge in tiny_llama_graph.edges.values():
        assert positions[edge.from_node] < positions[edge.to_node]


def test_topological_sort_loop(cyclic_graph):
    with pytest.raises(ValueError):
        topological_sort(cyclic_graph)


def compute_heights(graph: Graph):
    # here we use only the property that i comes before j for each edge (i, j)
    order = topological_sort(graph)
    heights: dict[int, int] = {}

    for i in range(len(order) - 1, -1, -1):
        node_id = order[i]
        max_height = max([heights[graph.edges[edge_id].to_node]
                          for edge_id in graph.outcoming_edges(node_id)], default=-1)
        heights[node_id] = max_height + 1

    return heights


@pytest.mark.skipif(not MODEL.exists(), reason="TinyLlama not downloaded")
def test_topological_sort_height_inversions(tiny_llama_graph):
    order = topological_sort(tiny_llama_graph)
    heights = compute_heights(tiny_llama_graph)

    for i in range(0, len(order) - 1):
        assert heights[order[i]] >= heights[order[i + 1]]


def test_add_node_explicit_id():
    graph = Graph()
    assert graph.add_node(Node("A", "Add"), node_id=10) == 10
    assert graph.add_node(Node("B", "Add")) == 11
    assert graph.add_node(Node("C", "Add"), node_id=5) == 5
    assert graph.add_node(Node("D", "Add")) == 12


def test_add_node_duplicate_id():
    graph = Graph()
    A = graph.add_node(Node("A", "Add"))
    with pytest.raises(KeyError):
        graph.add_node(Node("B", "Add"), node_id=A)
    assert graph.node(A).name == "A"


def test_parallel_edges_ports():
    graph = Graph()
    split = graph.add_node(Node("split", "Split"))
    add = graph.add_node(Node("add", "Add"))

    e1 = graph.add_edge(Edge(split, 1, add, 0, ("1", "?"), "f32"))
    e0 = graph.add_edge(Edge(split, 0, add, 1, ("1", "4"), "f32"))

    assert graph.outcoming_edges(split) == {e0, e1}
    assert graph.incoming_edges(add) == {e0, e1}
    assert (graph.edges[e1].from_port, graph.edges[e1].to_port) == (1, 0)
    assert (graph.edges[e0].from_port, graph.edges[e0].to_port) == (0, 1)
    assert graph.edges[e1].shape == ("1", "?")

    graph.remove_edge(e1)

    assert graph.outcoming_edges(split) == {e0}
    assert graph.incoming_edges(add) == {e0}
    assert (graph.edges[e0].from_port, graph.edges[e0].to_port) == (0, 1)
    assert e1 not in graph.edges


def test_remove_node():
    graph = Graph()

    A = graph.add_node(Node("A", "Add"))
    B = graph.add_node(Node("B", "Add"))
    C = graph.add_node(Node("C", "Add"))

    graph.add_edge(Edge(A, 0, B, 0))
    graph.add_edge(Edge(B, 0, C, 0))

    A_to_C = graph.add_edge(Edge(A, 0, C, 1))

    graph.remove_node(B)

    assert B not in graph.nodes
    assert set(graph.edges) == {A_to_C}
    assert graph.outcoming_edges(A) == {A_to_C}
    assert graph.incoming_edges(C) == {A_to_C}
    with pytest.raises(KeyError):
        graph.remove_node(B)
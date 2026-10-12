from parser import OpenVINOParser
from pathlib import Path
from graph import Graph, Node, Edge, topological_sort

import pytest

MODEL = Path(__file__).parent.parent / \
    "ov_TinyLlama_v1_1" / "openvino_model.xml"  # access the folder one level above


@pytest.fixture
def tiny_llama_graph():
    return OpenVINOParser().parse(MODEL)


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

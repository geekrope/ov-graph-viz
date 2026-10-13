from graph import Graph, Node, Edge
from transformation import ConstantHidingPass, HighlightingPass

import copy
import pytest


@pytest.fixture
def hidden_graph():
    graph = Graph()
    const1 = graph.add_node(Node("const1", "Constant"))
    const2 = graph.add_node(Node("const2", "Constant"))
    matmul = graph.add_node(Node("matmul", "MatMul"))
    add = graph.add_node(Node("add", "Add"))

    graph.add_edge(Edge(const1, 0, matmul, 0, ("4", "3"), "f32"))
    graph.add_edge(Edge(const2, 0, matmul, 1, ("3",), "f16"))
    graph.add_edge(Edge(add, 0, matmul, 2, ("1", "?", "4"), "i64"))

    ConstantHidingPass().apply(graph)
    return graph, const1, const2, matmul, add


@pytest.fixture
def shared_const_graph():
    graph = Graph()
    const_shared = graph.add_node(Node("const1", "Constant"))
    const_single = graph.add_node(Node("const2", "Constant"))
    matmul1 = graph.add_node(Node("matmul1", "MatMul"))
    matmul2 = graph.add_node(Node("matmul2", "MatMul"))

    graph.add_edge(Edge(const_shared, 0, matmul1, 0))
    graph.add_edge(Edge(const_shared, 0, matmul2, 0))
    graph.add_edge(Edge(const_single, 0, matmul2, 1))

    ConstantHidingPass().apply(graph)
    return graph, const_shared, const_single, matmul1, matmul2


@pytest.fixture
def invalid_const_graph():
    graph = Graph()
    relu = graph.add_node(Node("relu", "ReLU"))
    const = graph.add_node(Node("const", "Constant"))
    matmul = graph.add_node(Node("matmul", "MatMul"))

    graph.add_edge(Edge(relu, 0, const, 0))
    graph.add_edge(Edge(const, 0, matmul, 0))

    return graph, relu, const, matmul


def test_constant_hiding_in_degree(hidden_graph):
    graph, _, _, matmul, _ = hidden_graph
    meta = graph.node(matmul).meta
    assert graph.in_degree(matmul) + len(meta.hidden_inputs) == 3


def test_constant_hiding_consistency(hidden_graph):
    graph, _, _, matmul, add = hidden_graph
    [edge_id] = graph.incoming_edges(matmul)
    edge = graph.edges[edge_id]
    hidden = graph.node(matmul).meta.hidden_inputs

    assert set(hidden) == {0, 1}
    assert (hidden[0].shape, hidden[0].dtype) == (("4", "3"), "f32")
    assert (hidden[1].shape, hidden[1].dtype) == (("3",), "f16")
    assert (edge.from_node, edge.to_port) == (add, 2)
    assert (edge.shape, edge.dtype) == (("1", "?", "4"), "i64")


def test_constant_hiding_removal(hidden_graph):
    graph, const1, const2, _, _ = hidden_graph
    assert const1 not in graph.nodes and const2 not in graph.nodes


def test_constant_hiding_double_pass(hidden_graph):
    graph, _, _, _, _ = hidden_graph
    nodes_before = copy.deepcopy(graph.nodes)
    edges_before = copy.deepcopy(graph.edges)

    ConstantHidingPass().apply(graph)

    assert graph.nodes == nodes_before
    assert graph.edges == edges_before


def test_constant_hiding_shared(shared_const_graph):
    graph, _, _, matmul1, matmul2 = shared_const_graph

    assert graph.node(matmul1).meta.hidden_inputs[0].shared
    assert graph.node(matmul2).meta.hidden_inputs[0].shared
    assert not graph.node(matmul2).meta.hidden_inputs[1].shared


def test_constant_hiding_invalid(invalid_const_graph):
    graph, _, const, matmul = invalid_const_graph

    with pytest.warns(UserWarning, match="input ports"):
        ConstantHidingPass().apply(graph)

    assert const in graph.nodes
    assert graph.in_degree(matmul) == 1
    assert graph.node(matmul).meta.hidden_inputs == {}


@pytest.fixture
def node_name_inclusion_graph():
    graph = Graph()
    const = graph.add_node(Node("const", "Constant"))
    const_1 = graph.add_node(Node("const_1", "Constant"))
    matmul = graph.add_node(Node("matmul", "MatMul"))

    graph.add_edge(Edge(const, 0, matmul, 0))
    graph.add_edge(Edge(const_1, 0, matmul, 1))

    return graph, const, const_1, matmul


@pytest.fixture
def node_name_repeated_graph():
    graph = Graph()
    const = graph.add_node(Node("const", "Constant"))
    const_dupl = graph.add_node(Node("const", "Constant"))
    matmul = graph.add_node(Node("matmul", "MatMul"))

    graph.add_edge(Edge(const, 0, matmul, 0))
    graph.add_edge(Edge(const_dupl, 0, matmul, 1))

    return graph, const, const_dupl, matmul


def test_highlighting_exact_name_match(node_name_inclusion_graph):
    graph, const, const_1, _ = node_name_inclusion_graph

    HighlightingPass({"const"}).apply(graph)

    assert graph.node(const).meta.border_color == "red"
    assert graph.node(const_1).meta.border_color is None


def test_highlighting_multiple(node_name_inclusion_graph):
    graph, const, const_1, matmul = node_name_inclusion_graph

    HighlightingPass({"const", "const_1"}).apply(graph)

    assert graph.node(const).meta.border_color == "red"
    assert graph.node(const_1).meta.border_color == "red"
    assert graph.node(matmul).meta.border_color is None


def test_highlighting_repeated(node_name_repeated_graph):
    graph, const, const_dupl, matmul = node_name_repeated_graph

    HighlightingPass({"const"}).apply(graph)

    assert graph.node(const).meta.border_color == "red"
    assert graph.node(const_dupl).meta.border_color == "red"
    assert graph.node(matmul).meta.border_color is None
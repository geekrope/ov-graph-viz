from graph import Graph, Node, Edge
from renderer import GraphvizRenderer
from transformation import ConstantHidingPass, HighlightingPass

import copy
import pytest
import networkx as nx
import pydot
import re


# input -> matmul1 -> add -> matmul2 -> convert -> result1
#                         \-> result2
# weights (shared const) -> matmul1, matmul2; bias (scalar const) -> add
@pytest.fixture
def sample_graph():
    graph = Graph()
    ids: dict[str, int] = {}

    def add(name: str, node_type: str, attributes: dict[str, str] | None = None):
        ids[name] = graph.add_node(Node(name, node_type, attributes or {}))

    add("input", "Parameter", {"shape": "[1,?,4]", "element_type": "f32"})
    add("weights", "Constant", {"shape": "[4,4]", "element_type": "f32"})
    add("bias", "Constant", {"shape": "[]", "element_type": "f32"})
    add("matmul1", "MatMul", {"transpose_a": "false", "transpose_b": "true"})
    add("add", "Add", {"auto_broadcast": "numpy"})
    add("matmul2", "MatMul")
    add("convert", "Convert", {"destination_type": "f16"})
    add("result1", "Result")
    add("result2", "Result")

    graph.add_edge(Edge(ids["input"], 0, ids["matmul1"], 0, ("1", "?", "4"), "f32"))
    graph.add_edge(Edge(ids["weights"], 0, ids["matmul1"], 1, ("4", "4"), "f32"))
    graph.add_edge(Edge(ids["matmul1"], 0, ids["add"], 0, ("1", "?", "4"), "f32"))
    graph.add_edge(Edge(ids["bias"], 0, ids["add"], 1, (), "f32"))
    graph.add_edge(Edge(ids["add"], 0, ids["matmul2"], 0, ("1", "?", "4"), "f32"))
    graph.add_edge(Edge(ids["weights"], 0, ids["matmul2"], 1, ("4", "4"), "f32"))
    graph.add_edge(Edge(ids["add"], 0, ids["result2"], 0, ("1", "?", "4"), "f32"))
    graph.add_edge(Edge(ids["matmul2"], 0, ids["convert"], 0, ("1", "?", "4"), "f32"))
    graph.add_edge(Edge(ids["convert"], 0, ids["result1"], 0))

    return graph, ids


@pytest.fixture
def hidden_sample_graph(sample_graph):
    graph, ids = sample_graph
    original = copy.deepcopy(graph)
    ConstantHidingPass().apply(graph)
    return graph, original, ids


def build_graph(graph: Graph):
    dot_graph = GraphvizRenderer().build(graph)
    parsed = pydot.graph_from_dot_data(dot_graph.source)
    assert parsed is not None and len(
        parsed) == 1, "renderer produced invalid DOT"

    nx_repr = nx.nx_pydot.from_pydot(parsed[0])
    assert nx_repr.is_directed()
    assert isinstance(nx_repr, nx.MultiDiGraph)
    return nx_repr


def node_type(nx_graph: nx.MultiDiGraph, node: str) -> str:
    label = nx_graph.nodes[node]["label"].strip('"')  # '{Type|name: ...|...}'
    return label[1:].split("|")[0]


def check_structure(graph: Graph) -> None:
    nx_graph = build_graph(graph)

    assert set(nx_graph.nodes) == {str(node_id) for node_id in graph.nodes}
    for node_id, node in graph.nodes.items():
        assert node_type(nx_graph, str(node_id)) == node.node_type
        assert nx_graph.in_degree(str(node_id)) == graph.in_degree(node_id)
        assert nx_graph.out_degree(str(node_id)) == graph.out_degree(node_id)


def test_node_degrees(sample_graph):
    graph, _ = sample_graph
    check_structure(graph)


def test_node_degrees_hidden(hidden_sample_graph):
    graph, _, ids = hidden_sample_graph
    assert str(ids["weights"]) not in build_graph(graph).nodes
    check_structure(graph)


def test_node_coloring_highlighted(sample_graph):
    graph, ids = sample_graph
    highlighted = {"matmul1", "matmul2", "weights"}
    highlighted_ids = {str(ids[h]) for h in highlighted}
    graph = HighlightingPass(highlighted).apply(graph)
    nx_graph = build_graph(graph)

    for label, node in nx_graph.nodes.items():
        color = node["color"].strip('"')
        if label in highlighted_ids:
            assert color == "red"
        else:
            assert color == "black"


@pytest.mark.parametrize("shape, dtype, expected", [
    (("1", "?", "4"), "f32", "f32[1,?,4]"),
    (("1..8",), "i64", "i64[1..8]"),
    ((), "f32", "f32[]"),          # scalar
    (None, "f32", "f32[?]"),       # dynamic rank
    (("4",), None, "?[4]"),        # dynamic type
    (None, None, "?[?]"),
])
def test_format_tensor(shape, dtype, expected):
    assert GraphvizRenderer._format_tensor(shape, dtype) == expected


# TODO: vector constants match as well 
def test_hidden_ports_rendered(hidden_sample_graph):
    graph, original, _ = hidden_sample_graph
    nx_graph = build_graph(graph)
    for node_id in graph.nodes:
        label = nx_graph.nodes[str(node_id)]["label"]
        hidden = len(re.findall(r"\[(\d+)\]: ", label))
        assert nx_graph.in_degree(str(node_id)) + hidden == original.in_degree(node_id)


# TODO: match the number of nodes with expected when applying proxy pass
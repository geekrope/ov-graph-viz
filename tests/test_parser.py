from parser import OpenVINOParser
from graph import Graph

import pytest


@pytest.fixture
def tiny_model(tiny_model_path):
    return OpenVINOParser().parse(tiny_model_path)


def nodes_by_type(graph: Graph, type_name: str):
    return [k for k, v in graph.nodes.items() if v.node_type == type_name]


def edges_between(graph: Graph, a: int, b: int):
    return [e for e in graph.edges.values() if e.from_node == a and e.to_node == b]


def test_parser_edge_direction(tiny_model):
    x = nodes_by_type(tiny_model, "Parameter")[0]
    w = nodes_by_type(tiny_model, "Constant")[0]
    mm = nodes_by_type(tiny_model, "MatMul")[0]

    assert len(edges_between(tiny_model, w, mm)) > 0
    assert len(edges_between(tiny_model, x, mm)) > 0
    assert len(edges_between(tiny_model, mm, w)) == 0
    assert len(edges_between(tiny_model, mm, x)) == 0


def test_edge_meta(tiny_model):
    x = nodes_by_type(tiny_model, "Parameter")[0]
    mm = nodes_by_type(tiny_model, "MatMul")[0]
    add = nodes_by_type(tiny_model, "Add")[0]

    edge_mm_add = edges_between(tiny_model, mm, add)[0]
    edge_x_add = edges_between(tiny_model, x, add)[0]

    assert edge_mm_add.shape == ('1', '4')
    assert edge_mm_add.dtype == 'f32'
    assert edge_x_add.to_port == 1


def test_missing_edges(tiny_model):
    assert len(tiny_model.edges) == 6

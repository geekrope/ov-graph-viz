from parser import OpenVINOParser
from graph import Graph
from pathlib import Path
from conftest import MODEL

import pytest
import openvino as ov
import openvino.opset13 as ops
import numpy as np


@pytest.fixture
def tiny_model(tiny_model_path):
    return OpenVINOParser().parse(tiny_model_path)


# x[2,4] -> split -> (out 1 -> add in 0, out 0 -> add in 1) -> result
@pytest.fixture
def split_model(tmp_path):
    x = ops.parameter([2, 4], np.float32, name="x")
    split = ops.split(x, 0, 2, name="split")
    add = ops.add(split.output(1), split.output(0), name="add")
    res = ops.result(add, name="out")
    model = ov.Model([res], [x], "split")  # pyright: ignore[reportGeneralTypeIssues]

    path = tmp_path / "split.xml"
    ov.save_model(model, path, compress_to_fp16=False)
    return OpenVINOParser().parse(path)


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


def test_multi_output_ports(split_model):
    split = nodes_by_type(split_model, "Split")[0]
    add = nodes_by_type(split_model, "Add")[0]

    ports = {(e.from_port, e.to_port) for e in edges_between(split_model, split, add)}

    assert ports == {(1, 0), (0, 1)}


def test_handles_errors(tmp_path):
    parser = OpenVINOParser()

    garbled_xml = """<?xml version="1.0"?>
    <net name="broken" version="11">
            <layers>
                    <layer id="0" name="x" type="Parameter" version="opset1">
                            <data shape="1,4" element_type="f32" />
                            <output>
                                    <port id="0" precision="FP32">
                                            <dim>1</dim>
                                            <dim>4</dim>
                                    </port>
                            </output>
                    </layer>
                    <layer id="1" name="relu" type="Relu" version="opset1">
                            <input>
                                    <port id="0" precision="FP32">
                                            <dim>1</dim>
                                            <dim>4</dim>
    """

    garbled_path = Path(tmp_path) / "garbled.xml"
    garbled_path.write_text(garbled_xml)

    with pytest.raises(ValueError, match = "not supported"):
        parser.load_model(MODEL.with_suffix(".txt"))
    with pytest.raises(FileNotFoundError):
        parser.load_model(Path("./non_existent.xml"))
    with pytest.raises(SyntaxError):
        parser.load_model(garbled_path)

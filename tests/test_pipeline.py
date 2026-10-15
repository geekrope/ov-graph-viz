from graph import Graph, Node, NodeMetadata
from transformation import HighlightingPass, Pipeline, TransformationPass

import pytest


class RecordingPass(TransformationPass):
    def __init__(self, name: str, log: list[str]) -> None:
        self.name = name
        self.log = log

    def apply(self, graph: Graph) -> Graph:
        self.log.append(self.name)
        return graph


@pytest.fixture
def recording_pipeline():
    log: list[str] = []
    pipeline = Pipeline([
        ("b", RecordingPass("b", log)),
        ("a", RecordingPass("a", log)),
        ("c", RecordingPass("c", log)),
    ])
    return pipeline, log


def test_pipeline_order(recording_pipeline):
    pipeline, log = recording_pipeline

    pipeline.apply(Graph())

    assert log == ["b", "a", "c"]


def test_pipeline_applies_all_steps():
    graph = Graph()
    x = graph.add_node(Node("x", "ReLU"))
    y = graph.add_node(Node("y", "ReLU"))
    z = graph.add_node(Node("z", "ReLU"))

    Pipeline([
        ("highlight_x", HighlightingPass({"x"})),
        ("highlight_y", HighlightingPass({"y"})),
    ]).apply(graph)

    assert graph.node(x).meta.border_color == "red"
    assert graph.node(y).meta.border_color == "red"
    assert graph.node(z).meta == NodeMetadata()


def test_pipeline_disable(recording_pipeline):
    pipeline, log = recording_pipeline

    pipeline.disable("a")
    pipeline.apply(Graph())

    assert log == ["b", "c"]


def test_pipeline_enable(recording_pipeline):
    pipeline, log = recording_pipeline

    pipeline.disable("a")
    pipeline.enable("a")
    pipeline.apply(Graph())

    assert log == ["b", "a", "c"]


def test_pipeline_unknown_step(recording_pipeline):
    pipeline, _ = recording_pipeline

    with pytest.raises(KeyError):
        pipeline.disable("missing")
    with pytest.raises(KeyError):
        pipeline.enable("missing")


def test_pipeline_duplicate_names():
    log: list[str] = []

    with pytest.raises(ValueError, match="Duplicate"):
        Pipeline([
            ("a", RecordingPass("a", log)),
            ("a", RecordingPass("a", log)),
        ])

from parser import OpenVINOParser
from transformation import ConstantHidingPass, TruncationPass, HighlightingPass, ProxyPass, Pipeline
from renderer import GraphvizRenderer
from pathlib import Path

graph = OpenVINOParser().parse(Path("../ov_TinyLlama_v1_1/openvino_model.xml"))
pipeline = Pipeline([
    ("const_hiding", ConstantHidingPass()),
    ("truncation", TruncationPass(1500)),
    ("highlighting", HighlightingPass({"attention_mask", "__module.model.layers.1/aten::add/Add"})),
    ("proxy", ProxyPass(3)),
])
#pipeline.disable("truncation")
#pipeline.disable("proxy")
graph = pipeline.apply(graph)

GraphvizRenderer().render(graph, Path("../simple_graph.svg"))
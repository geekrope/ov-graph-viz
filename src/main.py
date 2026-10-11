from parser import OpenVINOParser
from transformation import ConstantHidingPass, TruncationPass
from renderer import GraphvizRenderer
from pathlib import Path

graph = OpenVINOParser().parse(Path("../ov_TinyLlama_v1_1/openvino_model.xml"))
transformation_passes = [ConstantHidingPass(), TruncationPass(1200)]
for transformation_pass in transformation_passes:
    graph = transformation_pass.apply(graph)

GraphvizRenderer().render(graph, Path("../simple_graph.svg"))
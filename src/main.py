import openvino as ov

from parser import parse_openvino


def load_openvino_model(xml_path: str) -> ov.Model:
    return ov.Core().read_model(xml_path)


model = load_openvino_model("../ov_TinyLlama_v1_1/openvino_model.xml")
for op in model.get_ordered_ops():
    print(op.inputs())
    print(op.outputs())
    print(op.get_type_name())

graph = parse_openvino(model)
print(graph)
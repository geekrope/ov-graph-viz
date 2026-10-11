import openvino as ov

from graph import Edge, Node, Graph
from typing import cast


def _shape(output: ov.Output) -> tuple[str, ...] | None:
    """
    Returns:
        None if the shape is inferred while computation, otherwise a tuple of dimensions
    """
    ps = output.get_partial_shape()
    if ps.rank.is_dynamic:
        return None
    return tuple(str(d) for d in ps)


def _dtype(output: ov.Output) -> str | None:
    """
    Returns:
        None if the type is inferred while computation, otherwise type name
    """
    t = output.get_element_type()
    return None if t.is_dynamic() else t.get_type_name()


# TODO: think if topological sort can do the job in one pass
def parse_openvino(model: ov.Model) -> Graph:
    graph = Graph()
    node_ids: dict[ov.Node, int] = {}
    operations: list[ov.Node] = model.get_ordered_ops()

    # first traversal to gather all the nodes
    for op in operations:
        attributes: dict[str, str] = { k: str(v) for k, v in op.get_attributes().items() }
        node = Node(op.get_friendly_name(), op.get_type_name(), attributes)
        node_ids[op] = graph.add_node(node)

    for op in operations:
        for inp in op.inputs():
            src = inp.get_source_output()
            edge = Edge(
                node_ids[inp.get_node()], 
                inp.get_index(),
                node_ids[op],
                src.get_index(),
                _shape(src),
                _dtype(src))

            graph.add_edge(edge)

    return graph
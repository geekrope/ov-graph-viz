import openvino as ov

from abc import ABC, abstractmethod
from pathlib import Path
from graph import Edge, Node, Graph


class ModelParser(ABC):
    @abstractmethod
    def parse(self, path: Path) -> Graph:
        pass


class OpenVINOParser(ModelParser):
    @staticmethod
    def _shape(output: ov.Output) -> tuple[str, ...] | None:
        """
        Returns:
            None if the shape is inferred while computation, otherwise a tuple of dimensions
        """
        ps = output.get_partial_shape()
        if ps.rank.is_dynamic:
            return None
        return tuple(str(d) for d in ps)

    @staticmethod
    def _dtype(output: ov.Output) -> str | None:
        """
        Returns:
            None if the type is inferred while computation, otherwise type name
        """
        t = output.get_element_type()
        return None if t.is_dynamic() else t.get_type_name()

    def load_model(self, path: Path) -> ov.Model:
        return ov.Core().read_model(path.absolute())

    def parse(self, path: Path) -> Graph:
        model = self.load_model(path)
        graph = Graph()
        node_ids: dict[ov.Node, int] = {}
        operations: list[ov.Node] = model.get_ordered_ops()

        for op in operations:
            attributes: dict[str, str] = {
                k: str(v) for k, v in op.get_attributes().items()}
            node = Node(op.get_friendly_name(), op.get_type_name(), attributes)
            node_ids[op] = graph.add_node(node)

        for op in operations:
            for inp in op.inputs():
                src = inp.get_source_output()
                edge = Edge(
                    node_ids[src.get_node()],
                    src.get_index(),
                    node_ids[op],
                    inp.get_index(),
                    OpenVINOParser._shape(src),
                    OpenVINOParser._dtype(src))

                graph.add_edge(edge)

        return graph
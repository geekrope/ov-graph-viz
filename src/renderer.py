import graphviz

from abc import ABC, abstractmethod
from pathlib import Path
from graph import Edge, Graph, Node

# TODO: read through
# TODO: node highlighting

NODE_FILLCOLOR_BY_TYPE = {
    "Parameter": "#EDEADE",  # Alabaster
    "Result": "#EDEADE",  # Alabaster
    "Constant": "#FFD54F",  # Amber
    "MatMul": "#90CAF9",  # Blue
    "Add": "#B39DDB",  # Purple
    "Multiply": "#B39DDB",  # Purple
    "Convert": "#E0E0E0",  # Grey
}
DEFAULT_FILLCOLOR = "white"

# Characters with special meaning inside a record label
_RECORD_SPECIAL_CHARS = "\\{}|<>\""


class GraphvizRenderer:
    NEW_LINE = r"\l"  # left-aligned line break

    @staticmethod
    def _escape(text: str) -> str:
        return "".join(
            "\\" + c if c in _RECORD_SPECIAL_CHARS else c for c in text)

    @staticmethod
    def _format_tensor(shape: tuple[str, ...] |
                       None, dtype: str | None) -> str:
        dtype_str = dtype if dtype is not None else "?"
        shape_str = "?" if shape is None else ",".join(shape)
        return f"{dtype_str}[{shape_str}]"

    def _node_label(self, node_id: int, node: Node) -> str:
        nl = self.NEW_LINE
        header = self._escape(node.node_type)
        info = f"name: {self._escape(node.name)}{nl}id: {node_id}{nl}"
        hidden_ports = "".join(f"[{k}]: {self._format_tensor(v.shape, v.dtype)}" for k, v in node.meta.hidden_inputs.items())
        attrs = "".join(
            f"{self._escape(k)}: {self._escape(v)}{nl}" for k, v in node.attributes.items())
        if attrs and hidden_ports:
            return f"{{{header}|{info}|{attrs}|{hidden_ports}}}"
        elif attrs:
            return f"{{{header}|{info}|{attrs}}}"
        else:
            return f"{{{header}|{info}|{hidden_ports}}}"

    def _edge_label(self, edge: Edge) -> str:
        tensor = self._format_tensor(edge.shape, edge.dtype)
        return f"out {edge.from_port} → in {edge.to_port}\n{tensor}"

    def build(self, graph: Graph) -> graphviz.Digraph:
        dg = graphviz.Digraph(format="svg")

        dg.graph_attr["rankdir"] = "TB"
        dg.graph_attr["ranksep"] = "0.2"
        dg.node_attr["style"] = "rounded,filled"
        dg.node_attr["shape"] = "record"
        dg.node_attr["fontsize"] = "10"
        dg.node_attr["fontname"] = "Tahoma"
        dg.node_attr["fillcolor"] = DEFAULT_FILLCOLOR
        dg.edge_attr["arrowsize"] = "0.5"
        dg.edge_attr["fontsize"] = "8"
        dg.edge_attr["fontname"] = "Tahoma"

        for node_id, node in graph.nodes.items():
            dg.node(
                str(node_id),
                label=self._node_label(node_id, node),
                fillcolor=NODE_FILLCOLOR_BY_TYPE.get(
                    node.node_type, DEFAULT_FILLCOLOR),
                id=f"node{node_id}",
            )

        for edge in graph.edges.values():
            dg.edge(str(edge.from_node), str(edge.to_node),
                    label=self._edge_label(edge))

        return dg

    def render(self, graph: Graph, path: Path) -> None:
        self.build(graph).render(outfile=path, cleanup=True)

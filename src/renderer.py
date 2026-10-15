import graphviz

from abc import ABC, abstractmethod
from pathlib import Path
from graph import Edge, Graph, Node, HiddenInput

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


class Renderer(ABC):
    @abstractmethod
    def render(self, graph: Graph, path: Path):
        pass


class GraphvizRenderer(Renderer):
    NEW_LINE = r"\l"  # left-aligned line break

    @staticmethod
    def _escape(text: str) -> str:
        return "".join(
            ("\\" + c) if c in _RECORD_SPECIAL_CHARS else c for c in text)

    @staticmethod
    def _format_tensor(shape: tuple[str, ...] |
                       None, dtype: str | None) -> str:
        dtype_str = dtype if dtype is not None else "?"
        shape_str = "?" if shape is None else ",".join(shape)
        return f"{dtype_str}[{shape_str}]"

    def _hidden_input_label(self, port: int, hidden: HiddenInput) -> str:
        label = f"[{port}]: {self._format_tensor(hidden.shape, hidden.dtype)}"
        if hidden.shared:
            label += f" (shared: \\<{hidden.id}\\>)"
        return label

    def _node_label(self, node_id: int, node: Node) -> str:
        nl = self.NEW_LINE
        header = self._escape(node.node_type)
        info = f"name: {self._escape(node.name)}{nl}id: {node_id}{nl}"
        hidden_ports = "".join(
            self._hidden_input_label(k, v)
            for k, v in node.meta.hidden_inputs.items()
        )
        attrs = "".join(
            f"{self._escape(k)}: {self._escape(v)}{nl}" for k, v in node.attributes.items())
        if attrs and hidden_ports:
            return f"{{{header}|{info}|{attrs}|{hidden_ports}}}"
        elif attrs:
            return f"{{{header}|{info}|{attrs}}}"
        elif hidden_ports:
            return f"{{{header}|{info}|{hidden_ports}}}"
        else:
            return f"{{{header}|{info}}}"

    def _edge_label(self, edge: Edge) -> str:
        tensor = self._format_tensor(edge.shape, edge.dtype)
        return f"out {edge.from_port} → in {edge.to_port}\n{tensor}"

    def _configure_graph(self) -> graphviz.Digraph:
        dg = graphviz.Digraph(format="svg")

        dg.graph_attr["rankdir"] = "TB"
        dg.graph_attr["ranksep"] = "0.2"
        dg.graph_attr["group"] = "main"

        dg.node_attr["shape"] = "record"
        dg.node_attr["fontsize"] = "10"
        dg.node_attr["fontname"] = "Tahoma"
        dg.node_attr["penwidth"] = "2"
        dg.node_attr["fillcolor"] = DEFAULT_FILLCOLOR

        dg.edge_attr["arrowsize"] = "0.5"
        dg.edge_attr["fontsize"] = "8"
        dg.edge_attr["fontname"] = "Tahoma"

        # acceleration
        # dg.graph_attr["splines"] = "line"
        # dg.graph_attr["nslimit"] = "2"
        return dg

    def build(self, graph: Graph) -> graphviz.Digraph:
        dg = self._configure_graph()

        for node_id, node in graph.nodes.items():
            fill_color = NODE_FILLCOLOR_BY_TYPE.get(
                node.node_type, DEFAULT_FILLCOLOR)
            dg.node(
                str(node_id),
                label=self._node_label(node_id, node),
                fillcolor=fill_color,
                color=node.meta.border_color,
                style=node.meta.style,
                id=f"node{node_id}",
            )

        for edge in graph.edges.values():
            src_node = graph.node(edge.from_node)
            src_id = edge.from_node
            dst_id = edge.to_node
            if src_node.meta.proxy_output:
                # node is created for the specific input
                name = f"{str(edge.from_node)}_{edge.to_node}_{edge.to_port}"
                fill_color = NODE_FILLCOLOR_BY_TYPE.get(
                    src_node.node_type, DEFAULT_FILLCOLOR)
                dg.node(
                    name,
                    label=self._node_label(src_id, src_node),
                    fillcolor=fill_color,
                    color=src_node.meta.border_color,
                    style="rounded,filled,dashed",
                    id=f"node{name}",
                )
                dg.edge(name, str(dst_id),
                        label=self._edge_label(edge))
            else:
                dg.edge(str(src_id), str(dst_id),
                        label=self._edge_label(edge))

        return dg

    def render(self, graph: Graph, path: Path) -> None:
        dg = self.build(graph)
        dg.save(path.with_suffix('.dot'))
        dg.render(outfile=path, cleanup=True)

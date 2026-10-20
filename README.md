# Model Graph Visualization

A small library and CLI that loads an OpenVINO model, converts it into a
framework-independent graph, applies a configurable sequence of transformation
passes and renders the result to SVG with Graphviz.

```
OpenVINO IR (.xml/.bin) -> parser -> Graph -> pass 1 -> ... -> pass N -> renderer -> SVG
```

## Getting started

Requirements: Python 3.13 (3.14 is not supported by the pinned dependencies)
and the Graphviz binaries (`dot` must be on `PATH`;
the `graphviz` Python package is only a wrapper around them).

```
git clone https://github.com/geekrope/ov-graph-viz.git
cd ov-graph-viz

# Graphviz: https://graphviz.org/download/
winget install graphviz        # Windows
sudo apt install graphviz      # Ubuntu / Debian

pip install -r requirements.txt
```

To run the tool, export a model as described in [Example models](#example-models)
and render it as shown in [Usage](#usage).

## Example models

Model files are not committed. They are exported with Optimum Intel into the
project root (the TinyLlama folder name is what the tests expect):

```
pip install "optimum[openvino]"
optimum-cli export openvino --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --weight-format fp16 ov_TinyLlama_v1_1
optimum-cli export openvino --model microsoft/resnet-18 --task image-classification --weight-format fp16 ov_resnet18
optimum-cli export openvino --model microsoft/speecht5_tts --model-kwargs '{\"vocoder\": \"microsoft/speecht5_hifigan\"}' --weight-format fp16 ov_speecht5
```

Without `--weight-format fp16`, `optimum-cli` compresses the weights to int8
for models larger than 1B parameters; this project uses the fp16 variant.
SpeechT5 is exported as several IR files; the examples use only the HiFi-GAN
vocoder, `ov_speecht5/openvino_vocoder.xml`.

| Model | Kind |
|---|---|
| TinyLlama-1.1B-Chat-v1.0 | LLM |
| ResNet-18 | CNN |
| HiFi-GAN vocoder | audio, mel spectrogram -> waveform |

## Usage

### CLI

```
python src/main.py list-passes
python src/main.py visualize --model MODEL.xml [--output OUT.svg] [--passes P1,P2,...] [pass options] [--timing]
```

- `--passes` is a comma-separated list; passes run in the given order. Without
  `--passes` the graph is rendered as is.
- `--timing` prints the time of parsing, of each pass and of rendering.
- `--output` defaults to `simple_graph.svg`.

Available passes (`list-passes`):

| Pass | Effect | Options |
|---|---|---|
| `hide-constants` | Removes Constant nodes and shows their shape/dtype inside the consumer node as [port]: dtype[shape]. Use to declutter weights and scalars. | - |
| `highlight` | Draws a red border around nodes with the given names (exact match, all duplicates). Use to find specific ops in a large graph. | `--highlight-names NAME[,NAME...]` |
| `truncate` | Keeps N nodes starting from position OFFSET in depth-wise as-late-as-possible topological order, i.e. a contiguous part of the network. Retains the connection with nodes left out of frame. Use to render large models quickly. Use `--truncate-size` = -1 to keep everything from offset to the end | `--truncate-size` N (default -1), `--truncate-offset` OFFSET (default 0) |
| `proxy` | Draws a dashed copy of a node next to each consumer instead of long edges when it has more than K consumers. Use for shared inputs (attention mask, positions). | `--max-consumers` K (default 3) |

Examples:

```
# no passes
python src/main.py visualize --model ov_resnet18\openvino_model.xml --output resnet18.svg

# hide constants and highlight a residual Add
python src/main.py visualize --model ov_resnet18\openvino_model.xml --output resnet18_passes.svg --passes hide-constants,highlight --highlight-names "__module.resnet.encoder.stages.0.layers.0/aten::add_/Add"

# a 500-node window of TinyLlama with proxies for shared inputs
python src/main.py visualize --model ov_TinyLlama_v1_1\openvino_model.xml --output tiny_llama.svg --passes hide-constants,truncate,proxy --truncate-size 500 --timing
```

Invalid arguments and unreadable models are reported as a CLI error (exit
code 2), e.g. an unknown pass name, a non-`.xml` file or a missing file.

### Python API

The modules live in `src/`; add it to `PYTHONPATH` (or run from `src/`).

```python
from pathlib import Path
from parser import OpenVINOParser
from transformation import Pipeline, ConstantHidingPass, HighlightingPass, TruncationPass, ProxyPass
from renderer import GraphvizRenderer

graph = OpenVINOParser().parse(Path("ov_TinyLlama_v1_1/openvino_model.xml"))

pipeline = Pipeline([
    ("hide", ConstantHidingPass()),
    ("window", TruncationPass(size=500, offset=0)),
    ("proxy", ProxyPass(max_consumers=3)),
    ("highlight", HighlightingPass({"__module.model.layers.0/aten::add/Add"})),
])
pipeline.disable("highlight")          # passes can be switched off by name
graph = pipeline.apply(graph)
print(pipeline.timings)                # seconds per enabled pass

GraphvizRenderer().render(graph, Path("tiny_llama.svg"))
```

A pass can also be applied directly: `graph = ConstantHidingPass().apply(graph)`.
Unlike the CLI, the API can use the same pass type several times with different
settings, as long as the step names differ.

## Examples

Generated SVGs are in [`examples/`](examples):

| File | Model | Passes | Nodes / edges |
|---|---|---|---|
| `resnet18.svg` | ResNet-18 | none | 155 / 162 |
| `resnet18_hide_const.svg` | ResNet-18 | `hide-constants` | 111 / 118 |
| `speecht5_vocoder.svg` | HiFi-GAN vocoder | none | 605 / 700 |
| `speecht5_vocoder_hide_const.svg` | HiFi-GAN vocoder | `hide-constants` | 424 / 467 |
| `tiny_llama_500_hide_const.svg` | TinyLlama | `hide-constants`, `truncate` (first 500 nodes) | 701 / 848 |
| `tiny_llama_500_hide_const_proxy.svg` | TinyLlama | `hide-constants`, `truncate` (first 500 nodes), `proxy` | 701 / 848 |
| `tiny_llama_proxy.svg` | TinyLlama, full graph | `proxy`, `highlight` (`Result_24221`) | 3167 / 3738 |

The examples were produced with:

```
python src/main.py visualize --model ov_resnet18\openvino_model.xml --output examples\resnet18.svg
python src/main.py visualize --model ov_resnet18\openvino_model.xml --output examples\resnet18_hide_const.svg --passes hide-constants
python src/main.py visualize --model ov_speecht5\openvino_vocoder.xml --output examples\speecht5_vocoder.svg
python src/main.py visualize --model ov_speecht5\openvino_vocoder.xml --output examples\speecht5_vocoder_hide_const.svg --passes hide-constants
```

TinyLlama:

```
python src/main.py visualize --model ov_TinyLlama_v1_1\openvino_model.xml --output examples\tiny_llama_proxy.svg --passes proxy,highlight --max-consumers 3 --highlight-names Result_24221 --timing
python src/main.py visualize --model ov_TinyLlama_v1_1\openvino_model.xml --output examples\tiny_llama_500_hide_const.svg --passes hide-constants,truncate --truncate-size 500
python src/main.py visualize --model ov_TinyLlama_v1_1\openvino_model.xml --output examples\tiny_llama_500_hide_const_proxy.svg --passes hide-constants,truncate,proxy --truncate-size 500 --max-consumers 3
```

The 500-node window renders as 701 nodes and 848 edges, because `truncate`
keeps the neighbours just outside the window.

## Architecture

| Module | Responsibility |
|---|---|
| `src/graph.py` | intermediate representation: `Graph`, `Node`, `Edge`, `NodeMetadata`, `topological_sort` |
| `src/parser.py` | `ModelParser` interface and `OpenVINOParser`; the only module that imports OpenVINO |
| `src/transformation.py` | `TransformationPass` interface, `Pipeline` and the passes |
| `src/renderer.py` | `Renderer` interface and `GraphvizRenderer` |
| `src/main.py` | CLI on top of the API; registry of passes for the CLI |

Passes and the renderer see only `Graph`, so they do not depend on OpenVINO,
and the parser does not know about passes or rendering.

Key design decisions:

- **Own node ids.** Nodes are stored in a dict under an integer id assigned by
  the graph, separate from the name, so duplicate names are fine.
- **Edges have ids and ports.** An edge stores the source node and output port,
  the destination node and input port, shape and dtype. Edges live in a dict
  by id, so several edges between the same pair of nodes are allowed.
  Per-node sets of incoming and outgoing edge ids make neighbour lookup and
  node/edge removal O(degree) instead of a scan over all edges; removing a node
  removes its edges, so the graph stays consistent.
- **Missing metadata is explicit.** A dynamic dimension is kept as OpenVINO
  prints it (`?`), a dynamic rank or dtype is `None`; the renderer shows both
  as `?`.
- **Visualization metadata is separate from the operation.** `NodeMetadata`
  holds everything the passes set for drawing (border color, style, hidden
  inputs, proxy flag); node attributes stay as they came from the model.
- **Passes modify the graph in place** and return it. This avoids copying a
  graph of thousands of nodes per pass; the OpenVINO model itself is never
  touched, because the graph holds no references to it.
- **ALAP topological order for truncation.** `topological_sort` is Kahn's
  algorithm run from the outputs, so every node is placed as late as possible,
  right before its first consumer. The depth (max of the children depths +1, for nodes with no output 0 by default)
  counting starts from the outputs. The nodes are processed in a BFS-type order and inserted at the beginning of the array, 
  which ensures that the depths are non-increasing. That guarantees that there are no depth inversions and the node
  is added directly before the deepest node among its children. Constants then fall into the same window as
  the operation that uses them.
- **Graphviz `dot`** gives the layered top-to-bottom layout. Nodes are record
  labels (type, name, id, attributes, hidden inputs); edges are labelled
  `out i -> in j` with the tensor type. Operation types without a configured
  color get a default one, so unknown types render as well.

### Custom pass: `proxy`

In TinyLlama some nodes (attention mask, position ids, rotary embedding
tables) feed every one of the 22 layers. `ProxyPass` marks nodes with more than `K` consumers, and the
renderer draws a dashed copy of such a node next to each consumer instead of a
long edge. The copy keeps the original id in its label, so it is still clear
which node it stands for. Proxies are drawn by the renderer only, so the graph
itself keeps the same nodes and edges.

Before and after: `tiny_llama_500_hide_const.svg` and
`tiny_llama_500_hide_const_proxy.svg` show the same 500-node window without
and with `proxy`. See also `tiny_llama_proxy.svg` for the full graph.

## Extensibility

**New transformation pass.** Subclass `TransformationPass` and implement
`apply(graph) -> Graph`:

```python
class MyPass(TransformationPass):
    def apply(self, graph: Graph) -> Graph:
        for node in graph.nodes.values():
            if node.node_type == "Softmax":
                node.meta.border_color = "blue"
        return graph
```

The pass can be used from the API right away; the parser and renderer do not
change as long as the pass works through the existing graph operations and
`NodeMetadata` fields. To expose it in the CLI, add an entry to `PASSES` in
`src/main.py` (a factory that builds the pass from the CLI arguments, a
description and its options) and, if needed, the option flags.

**New model format.** Subclass `ModelParser` and implement
`parse(path) -> Graph`, e.g. an `ONNXParser` that adds a `Node` per ONNX node
and an `Edge` per tensor connection. Passes and the renderer work on the
resulting `Graph` unchanged. The CLI currently always uses `OpenVINOParser`, so
it needs a choice of parser by file extension.

**New output format.** Subclass `Renderer` and implement `render(graph, path)`.

## Tests

```
pytest
```

Most tests use small synthetic graphs and run without any downloads. Tests on
the real model are skipped if `ov_TinyLlama_v1_1/openvino_model.xml` is missing:

- topological sort on the full graph (`tests/test_graph.py`);
- truncation window sizes (`tests/test_transformation.py`).

## Performance

Environment: Intel Core Ultra 7 255H, 32 GB RAM, Windows 11, Python 3.13.9,
OpenVINO 2026.2.0, Graphviz 16.1.0.

TinyLlama-1.1B-Chat-v1.0 (fp16): 3167 nodes, 3738 edges. After `hide-constants`:
1910 nodes, 2428 edges.

| Stage | Time, s |
|---|---|
| Model loading and IR construction | 0.14 |
| `proxy` (`--max-consumers 3`) | 0.001 |
| `highlight` (`Result_24221`) | 0.0002 |
| Rendering the full model to SVG (`proxy`, `highlight` applied) | 11.7 |
| `hide-constants` (full model) | 0.003 |
| `truncate` (`--truncate-size 500`, after `hide-constants`) | 0.003 |
| Rendering the 500-node window to SVG (`hide-constants`, `truncate`, `proxy` applied) | 3.5 |

Graph processing takes a fraction of a second; rendering with Graphviz takes
more than 98% of the total time and is the bottleneck.

The plot [`timings.pdf`](timings.pdf) illustrates how the processing time changes depending on the number of nodes

## Known limitations and potential improvements

- **No paginated view.** The whole (possibly truncated) graph is rendered into a
  single SVG; there is no way to split a large model into pages and navigate
  between them.
- **Truncation works by node count, not by layers.** `truncate` keeps N
  consecutive nodes in topological order, so the window boundary can cut through
  the middle of a transformer layer instead of aligning with layer boundaries.
- **Slow on large models.** Processing models of TinyLlama's size takes a long
  time. A possible improvement is to render the model layer by layer: a
  layer-aware truncation that cuts at transformer block boundaries would produce
  one small SVG per layer, which is faster to lay out and also gives a
  paginated view.
- **Unstable Graphviz layout.** On large graphs the `dot` layout drifts: the main
  computation flow does not stay in a straight vertical line and nodes shift
  sideways.
- **One configuration per pass type in the CLI.** Pass options are global
  flags (e.g. `--highlight-names`), so the CLI cannot run the same pass twice
  with different settings, such as two `highlight` passes with different node
  sets. For highlighting this is not a real restriction: the pass only sets the
  border color of nodes matched by name, so it commutes with every other pass,
  and two highlighting passes are equivalent to one with the union of the names.
  The Python API has no such restriction.
- **No pattern collapsing.** Repeated subgraphs are drawn node by node. A possible improvement
  is a pass that finds such patterns and replaces each occurrence with a single
  node, keeping its external inputs and outputs, so the graph becomes much
  smaller and easier to read.

## Test model (TinyLlama)

Model files are not committed. Export TinyLlama v1.1 to OpenVINO IR into the
project root (the folder name is what the tests expect):

```
pip install "optimum[openvino]"
optimum-cli export openvino --model TinyLlama/TinyLlama_v1.1 ov_TinyLlama_v1_1
```

`optimum-cli` compresses the weights to int8 by default for models larger than
1B parameters, which is the variant used in this project. Use
`--weight-format fp16` to get an fp16 model instead.

## Tests

```
pytest
```

Most tests use small synthetic graphs and run without any downloads. Tests on
the real model are skipped if `ov_TinyLlama_v1_1/openvino_model.xml` is missing:

- topological sort on the full graph (`tests/test_graph.py`);
- rendering of proxy nodes (`tests/test_renderer.py`).

## Performance

Environment: Intel Core Ultra 7 255H, 32 GB RAM, Windows 11, Python 3.13.9,
OpenVINO 2026.2.0, Graphviz 16.1.0.

TinyLlama v1.1 (int8): 3921 nodes, 4492 edges. After `hide-constants`:
2352 nodes, 2849 edges.

| Stage | Time, s |
|---|---|
| Model loading and IR construction | 0.21 |
| `hide-constants` | 0.003 |
| `proxy` (`--max-consumers 3`) | 0.002 |
| Rendering the full model to SVG (only `proxy` applied) | 11.0 |

Graph processing takes a fraction of a second; rendering with Graphviz takes
more than 98% of the total time and is the bottleneck.

## Known limitations

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

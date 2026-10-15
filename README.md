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

from parser import OpenVINOParser
from pathlib import Path

import pytest
import openvino as ov
import openvino.opset13 as ops
import numpy as np

MODEL = Path(__file__).parent.parent / \
    "ov_TinyLlama_v1_1" / "openvino_model.xml"  # access the folder one level above


@pytest.fixture(scope="session")
def tiny_llama_graph():
    return OpenVINOParser().parse(MODEL)


@pytest.fixture
def tiny_model_path(tmp_path):
    x = ops.parameter([1, 4], np.float32, name="x")
    w = ops.constant(np.ones((4, 4), np.float32), name="w")
    mm = ops.matmul(x, w, False, False, name="mm")
    add = ops.add(mm, x, name="add")
    relu = ops.relu(add, name="relu")
    res = ops.result(relu, name="out")
    model = ov.Model([res], [x], "tiny")  # pyright: ignore[reportGeneralTypeIssues]

    path = tmp_path / "tiny.xml"
    ov.save_model(model, path, compress_to_fp16=False)
    return path
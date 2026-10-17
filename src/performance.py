import matplotlib.pyplot as plt
import numpy as np
import time
import copy
import os

from pathlib import Path
from parser import OpenVINOParser
from renderer import GraphvizRenderer
from transformation import TruncationPass

tiny_ollama_path = Path("../ov_TinyLlama_v1_1/openvino_model.xml")
parsed_model = OpenVINOParser().parse(tiny_ollama_path)
svg_location = Path("./temp.svg")

ns = np.linspace(100, 1000, 10, dtype=int)
timings = np.zeros_like(ns, dtype=float)

for i, n in enumerate(ns):
    renderer_instance = GraphvizRenderer()
    truncated = TruncationPass(n).apply(copy.deepcopy(parsed_model))
    start = time.time()
    renderer_instance.render(truncated, svg_location, save_dot = False)
    timings[i] = time.time() - start

    print(f"complete: {n}")

plt.plot(ns, timings)
plt.title("Tiny Llama")
plt.xlabel("Number of nodes")
plt.ylabel("Time, s")
plt.savefig("../timings.pdf")
plt.show()

os.remove(svg_location)
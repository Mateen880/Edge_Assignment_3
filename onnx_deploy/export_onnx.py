"""
Export all five models to ONNX (opset 13, dynamic batch dimension).

  * FP32 / pruned models  -> standard float ONNX graphs. The structured-pruned graph
    has the reduced channel counts; the unstructured-pruned graph keeps zeros.
  * PTQ / QAT INT8 models -> QDQ-format ONNX (QuantizeLinear / DequantizeLinear with
    INT8 weights and the scales / zero-points learned in PyTorch); ONNX Runtime fuses
    these into INT8 kernels (QLinearConv, ...).

Each exported graph is checked with onnx.checker and compared with the PyTorch output.
Usage:  python onnx_deploy/export_onnx.py
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np
import onnx
import onnxruntime as ort
import torch

from common.data import load_test_tensors
from common.model_zoo import MODEL_NAMES, load_model
from common.paths import ONNX_DIR
from common.utils import file_size_mb


def onnx_path(name):
    return os.path.join(ONNX_DIR, f"resnet18_{name}.onnx")


def export(name, sample):
    model = load_model(name).eval()
    path = onnx_path(name)
    torch.onnx.export(model, sample[:1], path, opset_version=13, dynamo=False,
                      input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}})
    onnx.checker.check_model(onnx.load(path))

    # numerical check against PyTorch on a few test images
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = model(sample).numpy()
    out = sess.run(None, {"input": sample.numpy()})[0]
    agree = (out.argmax(1) == ref.argmax(1)).mean() * 100
    print(f"[ONNX] {name:<20} -> {os.path.relpath(path)}  ({file_size_mb(path):.2f} MB)  "
          f"max|diff| {np.abs(out - ref).max():.4f}  top-1 agreement {agree:.1f}%")


def main():
    X, _ = load_test_tensors()
    sample = X[:64]
    for name in MODEL_NAMES:
        export(name, sample)


if __name__ == "__main__":
    main()

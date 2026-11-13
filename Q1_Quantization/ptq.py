"""
Q1-2 : Post-Training Static Quantization (PTQ) of the trained FP32 ResNet-18.

Steps
  1. Load trained FP32 checkpoint.
  2. Fuse Conv+BN+ReLU / Conv+BN layers.
  3. Configure the quantization backend ('x86') and qconfig.
  4. Insert observers and calibrate on a subset of the training data.
  5. Convert to an INT8 model.
  6. Save the INT8 checkpoint (+ TorchScript) and run a deployment check on the test set.

Usage:  python Q1_Quantization/ptq.py [--calib-images 2000]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from Q1_Quantization.calibration import calibrate
from common.data import get_test_loader
from common.model_zoo import load_model
from common.paths import PTQ_CKPT, PTQ_JIT
from common.quant_utils import BACKEND, convert_to_int8, load_int8_model, prepare_ptq, save_int8_model
from common.utils import evaluate, file_size_mb, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib-images", type=int, default=2000)
    args = ap.parse_args()
    set_seed(0)

    fp32 = load_model("fp32")
    print(f"[PTQ] loaded FP32 model, backend = {BACKEND}")

    prepared = prepare_ptq(fp32)                   # fusion + qconfig + observers
    print("[PTQ] fused modules, e.g. layer1[0].conv1 ->", type(prepared.layer1[0].conv1))
    calibrate(prepared, num_images=args.calib_images)
    int8 = convert_to_int8(prepared)               # real INT8 kernels
    print("[PTQ] converted, e.g. layer1[0].conv1 ->", type(int8.layer1[0].conv1))

    save_int8_model(int8, PTQ_CKPT, PTQ_JIT)
    print(f"[PTQ] saved INT8 checkpoint {PTQ_CKPT} ({file_size_mb(PTQ_CKPT):.2f} MB)")

    # deployment check: reload from disk and evaluate
    deployed = load_int8_model(PTQ_CKPT, qat=False)
    acc = evaluate(deployed, get_test_loader(), "cpu")
    print(f"[PTQ] reloaded INT8 model test accuracy: {acc:.2f}%")


if __name__ == "__main__":
    main()

"""
Q1-3 : Quantization-Aware Training (QAT) of ResNet-18.

Steps
  1. Initialise from the pretrained (trained) FP32 checkpoint.
  2. Fuse layers (QAT fusion keeps BN trainable) and insert FakeQuantize modules.
  3. Fine-tune with fake quantization in the forward pass.
  4. Convert the QAT model into a real INT8 model.
  5. Export the INT8 checkpoint (+ TorchScript) and run a deployment check.

Usage:  python Q1_Quantization/qat.py [--epochs 2] [--lr 1e-3]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.ao.quantization import FakeQuantizeBase

from Q1_Quantization.qat_finetune import qat_finetune
from common.data import get_test_loader, get_train_loader
from common.model_zoo import load_model
from common.paths import QAT_CKPT, QAT_JIT
from common.quant_utils import BACKEND, convert_to_int8, load_int8_model, prepare_qat, save_int8_model
from common.utils import evaluate, file_size_mb, get_device, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    set_seed(42)
    device = get_device()

    fp32 = load_model("fp32")
    print(f"[QAT] initialised from FP32 checkpoint, backend = {BACKEND}, device = {device}")

    qat_model = prepare_qat(fp32)              # fusion + FakeQuantize insertion
    n_fq = sum(1 for m in qat_model.modules() if isinstance(m, FakeQuantizeBase))
    print(f"[QAT] inserted {n_fq} fake-quantization modules, e.g. layer1[0].conv1 ->",
          type(qat_model.layer1[0].conv1))

    test_loader = get_test_loader()
    qat_model = qat_finetune(qat_model, get_train_loader(), test_loader, device, args.epochs, args.lr)

    int8 = convert_to_int8(qat_model)
    print("[QAT] converted, e.g. layer1[0].conv1 ->", type(int8.layer1[0].conv1))
    save_int8_model(int8, QAT_CKPT, QAT_JIT)
    print(f"[QAT] saved INT8 checkpoint {QAT_CKPT} ({file_size_mb(QAT_CKPT):.2f} MB)")

    deployed = load_int8_model(QAT_CKPT, qat=True)
    acc = evaluate(deployed, test_loader, "cpu")
    print(f"[QAT] reloaded INT8 model test accuracy: {acc:.2f}%")


if __name__ == "__main__":
    main()

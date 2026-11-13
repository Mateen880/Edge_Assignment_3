"""Central place for all project paths (so every script uses the same locations)."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
MODELS_DIR = os.path.join(ROOT, "models")
ONNX_DIR = os.path.join(MODELS_DIR, "onnx")
RESULTS_DIR = os.path.join(ROOT, "results")

FP32_CKPT = os.path.join(MODELS_DIR, "resnet18_fp32.pth")
PTQ_CKPT = os.path.join(MODELS_DIR, "resnet18_ptq_int8.pth")
PTQ_JIT = os.path.join(MODELS_DIR, "resnet18_ptq_int8_torchscript.pt")
QAT_CKPT = os.path.join(MODELS_DIR, "resnet18_qat_int8.pth")
QAT_JIT = os.path.join(MODELS_DIR, "resnet18_qat_int8_torchscript.pt")
UNSTRUCT_CKPT = os.path.join(MODELS_DIR, "resnet18_unstructured_pruned.pth")
STRUCT_CKPT = os.path.join(MODELS_DIR, "resnet18_structured_pruned.pth")

for _d in (DATA_DIR, MODELS_DIR, ONNX_DIR, RESULTS_DIR):
    os.makedirs(_d, exist_ok=True)

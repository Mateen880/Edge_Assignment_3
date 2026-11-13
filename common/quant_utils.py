"""Quantization helpers shared by the PTQ, QAT, inference and export scripts."""
import copy

import torch
import torch.ao.quantization as tq

from common.models import ResNet18

# Quantization backend for x86 CPUs (uses FBGEMM / oneDNN kernels).
BACKEND = "x86"


def configure_backend():
    """Select the INT8 kernel backend used by the quantized operators."""
    torch.backends.quantized.engine = BACKEND


def prepare_ptq(fp32_model):
    """Fuse + attach static-quantization qconfig + insert observers (eval mode)."""
    configure_backend()
    m = copy.deepcopy(fp32_model).cpu().eval()
    m.fuse_model(qat=False)                       # Conv+BN+ReLU / Conv+BN fusion
    m.qconfig = tq.get_default_qconfig(BACKEND)   # per-channel weights, histogram activations
    tq.prepare(m, inplace=True)                   # inserts observers
    return m


def prepare_qat(fp32_model):
    """Fuse (QAT-style) + attach QAT qconfig + insert fake-quantization modules (train mode)."""
    configure_backend()
    m = copy.deepcopy(fp32_model).cpu().train()
    m.fuse_model(qat=True)                            # keeps BN trainable (ConvBnReLU2d QAT modules)
    m.qconfig = tq.get_default_qat_qconfig(BACKEND)   # FakeQuantize for weights and activations
    tq.prepare_qat(m, inplace=True)                   # inserts FakeQuantize modules
    return m


def convert_to_int8(prepared_model):
    """Replace observed / fake-quantized float modules by real INT8 modules."""
    m = copy.deepcopy(prepared_model).cpu().eval()
    return tq.convert(m, inplace=False)


def int8_skeleton(qat=False):
    """Empty INT8 ResNet-18 with the correct module structure, used to load INT8 state_dicts."""
    base = ResNet18()
    prepared = prepare_qat(base) if qat else prepare_ptq(base)
    return convert_to_int8(prepared.eval())


def load_int8_model(path, qat=False):
    configure_backend()
    model = int8_skeleton(qat)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.eval()


def save_int8_model(model, ckpt_path, jit_path):
    """Save INT8 state_dict and a self-contained TorchScript file for deployment."""
    torch.save(model.state_dict(), ckpt_path)
    scripted = torch.jit.trace(model, torch.randn(1, 3, 32, 32))
    torch.jit.save(scripted, jit_path)

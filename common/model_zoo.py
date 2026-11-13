"""Single entry point to load any of the five deployable models by name."""
import torch

from common import paths
from common.models import ResNet18
from common.quant_utils import configure_backend, load_int8_model

MODEL_NAMES = ("fp32", "ptq_int8", "qat_int8", "unstructured_pruned", "structured_pruned")
ALIASES = {"ptq": "ptq_int8", "qat": "qat_int8", "unstructured": "unstructured_pruned",
           "structured": "structured_pruned"}

CKPT_PATHS = {
    "fp32": paths.FP32_CKPT,
    "ptq_int8": paths.PTQ_CKPT,
    "qat_int8": paths.QAT_CKPT,
    "unstructured_pruned": paths.UNSTRUCT_CKPT,
    "structured_pruned": paths.STRUCT_CKPT,
}


def _load_float(path):
    """Float checkpoints store the architecture config (mid_channels) next to the weights,
    so the structured-pruned (physically smaller) network is rebuilt exactly."""
    ckpt = torch.load(path, map_location="cpu")
    model = ResNet18(mid_channels=ckpt["mid_channels"])
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()


def load_model(name):
    name = ALIASES.get(name, name)
    if name not in MODEL_NAMES:
        raise ValueError(f"unknown model '{name}', choose from {MODEL_NAMES}")
    path = CKPT_PATHS[name]
    if name == "ptq_int8":
        return load_int8_model(path, qat=False)
    if name == "qat_int8":
        return load_int8_model(path, qat=True)
    configure_backend()
    return _load_float(path)

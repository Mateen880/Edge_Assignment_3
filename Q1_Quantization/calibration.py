"""
PTQ calibration pipeline.

Feeds a subset of the CIFAR-10 *training* set through a model that has observers
inserted (torch.ao.quantization.prepare). The observers record activation ranges
which are later used to compute the INT8 scale / zero-point of every layer.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from common.data import get_calibration_loader


@torch.no_grad()
def calibrate(prepared_model, num_images=2000, batch_size=100, seed=0):
    """Run the calibration subset through the observed model (forward passes only)."""
    loader = get_calibration_loader(num_images, batch_size, seed)
    prepared_model.eval()
    seen = 0
    for x, _ in loader:
        prepared_model(x)
        seen += x.size(0)
    print(f"[Calibration] passed {seen} training images through the observers "
          f"({len(loader)} batches)")
    return prepared_model

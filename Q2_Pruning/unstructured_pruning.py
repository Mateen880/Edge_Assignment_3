"""
Q2-2 : Iterative unstructured (magnitude) pruning of ResNet-18.

  * Global L1-magnitude pruning over the weights of every Conv2d and Linear layer.
  * Iterative schedule: `--iterations` rounds, each round prunes a fraction of the
    remaining weights so that the final sparsity equals `--target-sparsity`,
    followed by fine-tuning (masks keep the pruned weights at zero).
  * Masks are finally made permanent with prune.remove(), giving a plain dense
    state_dict with zeros (no weight_orig / weight_mask buffers).

Usage:  python Q2_Pruning/unstructured_pruning.py [--target-sparsity 0.8] [--iterations 4]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from Q2_Pruning.finetune import finetune
from common.data import get_test_loader, get_train_loader
from common.model_zoo import load_model
from common.paths import UNSTRUCT_CKPT
from common.utils import compressed_size_mb, evaluate, file_size_mb, get_device, set_seed, weight_sparsity


def prunable_params(model):
    """(module, 'weight') pairs for all convolutional and fully connected layers."""
    return [(m, "weight") for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]


def masked_sparsity(model):
    """Sparsity computed from the active pruning masks."""
    zeros = total = 0
    for m, _ in prunable_params(model):
        mask = m.weight_mask if hasattr(m, "weight_mask") else torch.ones_like(m.weight)
        zeros += int((mask == 0).sum())
        total += mask.numel()
    return zeros / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-sparsity", type=float, default=0.8)
    ap.add_argument("--iterations", type=int, default=4)
    ap.add_argument("--ft-epochs", type=int, default=1, help="fine-tuning epochs per iteration")
    ap.add_argument("--lr", type=float, default=5e-3)
    args = ap.parse_args()
    set_seed(42)
    device = get_device()

    model = load_model("fp32")
    train_loader, test_loader = get_train_loader(), get_test_loader()
    print(f"[Unstructured] original FP32 test accuracy: {evaluate(model, test_loader, device):.2f}%")

    # fraction of the *remaining* weights removed each round -> geometric schedule
    per_iter = 1 - (1 - args.target_sparsity) ** (1 / args.iterations)
    print(f"[Unstructured] target sparsity {args.target_sparsity:.0%} in {args.iterations} "
          f"iterations ({per_iter:.2%} of remaining weights per iteration)")

    for it in range(1, args.iterations + 1):
        prune.global_unstructured(prunable_params(model), pruning_method=prune.L1Unstructured,
                                  amount=per_iter)
        acc_pruned = evaluate(model, test_loader, device)
        print(f"[Unstructured] iteration {it}: sparsity {masked_sparsity(model):.2%}  "
              f"test_acc after pruning {acc_pruned:.2f}%")
        model, acc = finetune(model, train_loader, test_loader, device, args.ft_epochs, args.lr,
                              tag=f"Unstructured it{it}")

    # make pruning permanent: weight = weight_orig * mask, buffers removed
    for m, name in prunable_params(model):
        prune.remove(m, name)
    model = model.cpu().eval()
    assert not any("mask" in k for k in model.state_dict())

    torch.save({"state_dict": model.state_dict(), "mid_channels": model.mid_channels,
                "sparsity": weight_sparsity(model)}, UNSTRUCT_CKPT)
    print(f"[Unstructured] permanent sparsity {weight_sparsity(model):.2%}; saved {UNSTRUCT_CKPT} "
          f"({file_size_mb(UNSTRUCT_CKPT):.2f} MB, zlib-compressed {compressed_size_mb(UNSTRUCT_CKPT):.2f} MB)")

    deployed = load_model("unstructured_pruned")
    print(f"[Unstructured] reloaded model test accuracy: {evaluate(deployed, test_loader):.2f}%")


if __name__ == "__main__":
    main()

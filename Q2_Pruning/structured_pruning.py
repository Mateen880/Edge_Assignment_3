"""
Q2-3 : Structured channel pruning of ResNet-18 with architecture reconstruction.

Channel selection
  Inside every BasicBlock the output channels of conv1 (= input channels of conv2)
  are ranked by the L1-norm of their filters; the lowest `--prune-ratio` are removed.

Valid tensor dimensions
  Only the block-internal channels are pruned. The block's output channels feed the
  residual addition and the next block, so they are left unchanged; therefore the
  following tensors are sliced consistently:
      conv1.weight[out]   bn1.{weight,bias,running_mean,running_var}[out]
      conv2.weight[:, in]
  and every shortcut / next layer still receives the shape it expects.

Reconstruction
  A new ResNet18(mid_channels=[...]) is built with physically smaller Conv2d / BN
  layers and the kept weights are copied in, so the model runs without any masks or
  zero-valued channels. The pruned model is then fine-tuned.

Usage:  python Q2_Pruning/structured_pruning.py [--prune-ratio 0.5] [--ft-epochs 3]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from Q2_Pruning.finetune import finetune
from common.data import get_test_loader, get_train_loader
from common.model_zoo import load_model
from common.models import ResNet18
from common.paths import STRUCT_CKPT
from common.utils import count_params, evaluate, file_size_mb, get_device, set_seed


def channel_importance(conv):
    """L1-norm of every output filter of a conv layer."""
    return conv.weight.detach().abs().sum(dim=(1, 2, 3))


def select_channels(model, prune_ratio):
    """Indices of the channels to KEEP for every BasicBlock (sorted)."""
    keep = []
    for blk in model.blocks():
        imp = channel_importance(blk.conv1)
        n_keep = max(1, int(round(imp.numel() * (1 - prune_ratio))))
        keep.append(torch.sort(torch.topk(imp, n_keep).indices).values)
    return keep


@torch.no_grad()
def reconstruct(model, keep):
    """Build a smaller ResNet-18 and copy the surviving weights into it."""
    small = ResNet18(mid_channels=[len(k) for k in keep])

    # layers that are not touched by pruning are copied as-is
    for name in ("conv1", "bn1", "fc"):
        getattr(small, name).load_state_dict(getattr(model, name).state_dict())

    for old, new, idx in zip(model.blocks(), small.blocks(), keep):
        new.conv1.weight.copy_(old.conv1.weight[idx])                  # remove output filters
        for attr in ("weight", "bias", "running_mean", "running_var"):
            getattr(new.bn1, attr).copy_(getattr(old.bn1, attr)[idx])  # matching BN channels
        new.bn1.num_batches_tracked.copy_(old.bn1.num_batches_tracked)
        new.conv2.weight.copy_(old.conv2.weight[:, idx])               # remove input channels
        new.bn2.load_state_dict(old.bn2.state_dict())                  # output channels unchanged
        if old.downsample is not None:
            new.downsample.load_state_dict(old.downsample.state_dict())
    return small.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prune-ratio", type=float, default=0.5,
                    help="fraction of block-internal channels removed in every BasicBlock")
    ap.add_argument("--ft-epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-2)
    args = ap.parse_args()
    set_seed(42)
    device = get_device()

    model = load_model("fp32")
    train_loader, test_loader = get_train_loader(), get_test_loader()
    p_orig, _ = count_params(model)
    print(f"[Structured] original model: {p_orig / 1e6:.2f} M params, "
          f"test_acc {evaluate(model, test_loader, device):.2f}%")

    keep = select_channels(model, args.prune_ratio)
    small = reconstruct(model.cpu(), keep)
    p_new, _ = count_params(small)
    for i, (blk_old, k) in enumerate(zip(model.blocks(), keep)):
        print(f"[Structured] block {i}: conv1 out / conv2 in channels {blk_old.conv1.out_channels} -> {len(k)}")
    print(f"[Structured] reconstructed model: {p_new / 1e6:.2f} M params "
          f"({100 * (1 - p_new / p_orig):.1f}% fewer)")

    # sanity check: the reconstructed network runs with the reduced shapes
    out = small(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 10)
    print(f"[Structured] test_acc right after pruning (before fine-tuning): "
          f"{evaluate(small, test_loader, 'cpu'):.2f}%")

    small, _ = finetune(small, train_loader, test_loader, device, args.ft_epochs, args.lr, tag="Structured")
    small = small.cpu().eval()
    torch.save({"state_dict": small.state_dict(), "mid_channels": small.mid_channels,
                "prune_ratio": args.prune_ratio}, STRUCT_CKPT)
    print(f"[Structured] saved {STRUCT_CKPT} ({file_size_mb(STRUCT_CKPT):.2f} MB)")

    deployed = load_model("structured_pruned")
    print(f"[Structured] reloaded model test accuracy: {evaluate(deployed, test_loader):.2f}%")


if __name__ == "__main__":
    main()

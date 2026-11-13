"""
Q1-1 / Q2-1 : Train the baseline FP32 ResNet-18 on CIFAR-10 and save the checkpoint.

The network is initialised from ImageNet-pretrained ResNet-18 weights (new 10-class
head) and fine-tuned on CIFAR-10. The same FP32 checkpoint is the starting point for
PTQ, QAT, unstructured pruning and structured pruning.

Usage:  python Q1_Quantization/train_baseline.py [--epochs 6] [--lr 0.01]
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from common.data import get_test_loader, get_train_loader
from common.models import build_resnet18
from common.paths import FP32_CKPT
from common.utils import evaluate, get_device, set_seed, train_one_epoch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"[Baseline] device={device}  epochs={args.epochs}  lr={args.lr}")

    train_loader = get_train_loader(args.batch_size)
    test_loader = get_test_loader()

    model = build_resnet18(pretrained_imagenet=True).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                weight_decay=5e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train_loader))

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device, scheduler, epoch)
        te_acc = evaluate(model, test_loader, device)
        print(f"[Baseline] epoch {epoch}/{args.epochs}  train_loss {tr_loss:.4f}  "
              f"train_acc {tr_acc:.2f}%  test_acc {te_acc:.2f}%  ({time.time() - t0:.0f}s)", flush=True)
        if te_acc > best_acc:
            best_acc = te_acc
            torch.save({"state_dict": model.cpu().state_dict(),
                        "mid_channels": model.mid_channels,
                        "test_acc": te_acc}, FP32_CKPT)
            model.to(device)
    print(f"[Baseline] best FP32 test accuracy {best_acc:.2f}%  -> saved {FP32_CKPT}")


if __name__ == "__main__":
    main()

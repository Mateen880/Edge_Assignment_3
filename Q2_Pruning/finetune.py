"""
Fine-tuning pipeline used by both pruning approaches to recover accuracy.

For unstructured pruning the model still carries torch.nn.utils.prune masks while it is
fine-tuned (weight = weight_orig * weight_mask), so pruned weights stay exactly zero.
For structured pruning the model is already physically smaller, so plain training is used.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from common.utils import evaluate, train_one_epoch


def finetune(model, train_loader, test_loader, device, epochs=1, lr=5e-3, tag="FT"):
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                                weight_decay=5e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader))
    acc = None
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device, scheduler, epoch)
        acc = evaluate(model, test_loader, device)
        print(f"[{tag}] fine-tune epoch {epoch}/{epochs}  train_loss {loss:.4f}  "
              f"train_acc {tr_acc:.2f}%  test_acc {acc:.2f}%  ({time.time() - t0:.0f}s)", flush=True)
    return model, acc

"""Training / evaluation / measurement helpers shared by all scripts."""
import csv
import os
import random
import time
import zlib

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed=42):
    """Make runs reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, loader, optimizer, device, scheduler=None, epoch=0, log_every=100):
    """Standard supervised epoch. `scheduler` (if given) is stepped every iteration."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    total, correct, loss_sum = 0, 0, 0.0
    t0 = time.time()
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        loss_sum += loss.item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
        if (i + 1) % log_every == 0:
            print(f"  epoch {epoch} iter {i + 1:4d}/{len(loader)}  loss {loss_sum / total:.4f}  "
                  f"acc {100 * correct / total:.2f}%  ({time.time() - t0:.0f}s)", flush=True)
    return loss_sum / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, device="cpu", return_preds=False):
    """Top-1 accuracy on a loader. Optionally return all predictions."""
    model.eval()
    correct, total, preds = 0, 0, []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        p = model(x).argmax(1)
        correct += (p == y).sum().item()
        total += y.size(0)
        if return_preds:
            preds.append(p.cpu())
    acc = 100.0 * correct / total
    return (acc, torch.cat(preds)) if return_preds else acc


def file_size_mb(path):
    return os.path.getsize(path) / 1e6


def compressed_size_mb(path):
    """Size of the checkpoint after lossless (zlib) compression - shows the gain from sparsity."""
    with open(path, "rb") as f:
        return len(zlib.compress(f.read(), 6)) / 1e6


def count_params(model):
    """(total parameters, non-zero parameters) over float weight tensors."""
    total, nonzero = 0, 0
    for p in model.parameters():
        total += p.numel()
        nonzero += int((p != 0).sum())
    return total, nonzero


def weight_sparsity(model):
    """Fraction of zero weights in Conv2d and Linear layers."""
    zeros, total = 0, 0
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            zeros += int((m.weight == 0).sum())
            total += m.weight.numel()
    return zeros / max(total, 1)


def write_csv(path, rows):
    """Write a list of dicts to CSV (column order taken from the first row)."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def print_table(rows, cols):
    """Pretty-print selected columns of a list of dicts as an aligned console table."""
    widths = [max(len(c), *(len(str(r[c])) for r in rows)) for c in cols]
    print("  ".join(c.rjust(w) for c, w in zip(cols, widths)))
    for r in rows:
        print("  ".join(str(r[c]).rjust(w) for c, w in zip(cols, widths)))

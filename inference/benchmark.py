"""
Automated benchmarking of the PyTorch models on the CIFAR-10 test set (CPU).

  --group q1 : FP32, PTQ-INT8, QAT-INT8
  --group q2 : FP32 (original), unstructured-pruned, structured-pruned
  --group all: both (default)

For every model it measures: top-1 accuracy, checkpoint size (raw and zlib-compressed),
parameter count / sparsity, single-image latency and batched throughput.

Outputs (results/):
  q1_benchmark.csv, q1_predictions.csv   (per-image predictions of each Q1 model)
  q2_benchmark.csv, q2_predictions.csv   (per-image predictions of each Q2 model)
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from common.data import load_test_tensors
from common.model_zoo import CKPT_PATHS, load_model
from common.models import CIFAR10_CLASSES
from common.paths import RESULTS_DIR
from common.utils import print_table, write_csv, compressed_size_mb, count_params, file_size_mb, weight_sparsity

GROUPS = {
    "q1": ("fp32", "ptq_int8", "qat_int8"),
    "q2": ("fp32", "unstructured_pruned", "structured_pruned"),
}
NUM_THREADS = 4


@torch.no_grad()
def predict_all(model, X, batch_size=250):
    return torch.cat([model(X[i:i + batch_size]).argmax(1) for i in range(0, len(X), batch_size)])


@torch.no_grad()
def latency_ms(model, X, n=300, warmup=30):
    """Mean / p95 latency of single-image inference (batch size 1)."""
    for i in range(warmup):
        model(X[i:i + 1])
    times = []
    for i in range(n):
        t0 = time.perf_counter()
        model(X[i:i + 1])
        times.append((time.perf_counter() - t0) * 1e3)
    return float(np.mean(times)), float(np.percentile(times, 95))


@torch.no_grad()
def throughput(model, X, batch_size=250):
    """Images per second over the whole test set (batch size 250)."""
    model(X[:batch_size])  # warm-up
    t0 = time.perf_counter()
    predict_all(model, X, batch_size)
    return len(X) / (time.perf_counter() - t0)


def benchmark_group(group, X, Y):
    rows, preds = [], {}
    for name in GROUPS[group]:
        print(f"[Benchmark-{group}] {name} ...", flush=True)
        model = load_model(name).eval()
        p = predict_all(model, X)
        preds[name] = p.numpy()
        lat, p95 = latency_ms(model, X)
        thr = throughput(model, X)
        row = {
            "model": name,
            "test_accuracy_%": round(100 * (p == Y).float().mean().item(), 2),
            "checkpoint_MB": round(file_size_mb(CKPT_PATHS[name]), 2),
            "zlib_compressed_MB": round(compressed_size_mb(CKPT_PATHS[name]), 2),
            "latency_bs1_mean_ms": round(lat, 3),
            "latency_bs1_p95_ms": round(p95, 3),
            "throughput_img_per_s": round(thr, 1),
        }
        if "int8" in name:
            row.update({"precision": "INT8", "params_M": "", "weight_sparsity_%": ""})
        else:
            total, _ = count_params(model)
            row.update({"precision": "FP32", "params_M": round(total / 1e6, 3),
                        "weight_sparsity_%": round(100 * weight_sparsity(model), 2)})
        rows.append(row)
        print("   ", row, flush=True)

    base = rows[0]
    for r in rows:   # metrics relative to the FP32 baseline (first row)
        r["size_reduction_x"] = round(base["checkpoint_MB"] / r["checkpoint_MB"], 2)
        r["latency_speedup_x"] = round(base["latency_bs1_mean_ms"] / r["latency_bs1_mean_ms"], 2)
        r["throughput_speedup_x"] = round(r["throughput_img_per_s"] / base["throughput_img_per_s"], 2)
        r["accuracy_drop_%"] = round(base["test_accuracy_%"] - r["test_accuracy_%"], 2)
        r["device"] = f"CPU ({NUM_THREADS} threads)"

    y = Y.numpy()
    pred_cols = {"image_index": np.arange(len(y)), "true_label": y,
                 "true_class": [CIFAR10_CLASSES[i] for i in y]}
    for name, p in preds.items():
        pred_cols[f"{name}_pred"] = p
        pred_cols[f"{name}_pred_class"] = [CIFAR10_CLASSES[i] for i in p]
        pred_cols[f"{name}_correct"] = (p == y).astype(int)
    pred_rows = [dict(zip(pred_cols, vals)) for vals in zip(*pred_cols.values())]

    write_csv(os.path.join(RESULTS_DIR, f"{group}_benchmark.csv"), rows)
    write_csv(os.path.join(RESULTS_DIR, f"{group}_predictions.csv"), pred_rows)
    print(f"\n[Benchmark-{group}] results written to results/{group}_benchmark.csv and "
          f"results/{group}_predictions.csv")
    cols = ["model", "precision", "test_accuracy_%", "checkpoint_MB", "zlib_compressed_MB",
            "latency_bs1_mean_ms", "throughput_img_per_s", "size_reduction_x", "latency_speedup_x"]
    print_table(rows, cols)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=("q1", "q2", "all"), default="all")
    args = ap.parse_args()
    torch.set_num_threads(NUM_THREADS)
    torch.manual_seed(0)

    X, Y = load_test_tensors()
    for g in (("q1", "q2") if args.group == "all" else (args.group,)):
        benchmark_group(g, X, Y)


if __name__ == "__main__":
    main()

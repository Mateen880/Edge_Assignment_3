"""
CPU inference with ONNX Runtime for the five exported ONNX models (no PyTorch model code needed).

Modes
  python onnx_deploy/onnx_inference.py                              # benchmark all -> results/onnx_benchmark.csv
  python onnx_deploy/onnx_inference.py --model qat_int8 --num 10    # classify test images
  python onnx_deploy/onnx_inference.py --model fp32 --image cat.png # classify an image file
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import onnxruntime as ort
from PIL import Image

from common.paths import ONNX_DIR, RESULTS_DIR
from common.utils import write_csv

MODEL_NAMES = ("fp32", "ptq_int8", "qat_int8", "unstructured_pruned", "structured_pruned")
CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(1, 3, 1, 1)
STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32).reshape(1, 3, 1, 1)
NUM_THREADS = 4


def session(name):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = NUM_THREADS
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    path = os.path.join(ONNX_DIR, f"resnet18_{name}.onnx")
    return ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"]), path


def preprocess(uint8_nhwc):
    """uint8 HxWx3 images -> normalised float32 NCHW (pure NumPy preprocessing)."""
    x = uint8_nhwc.astype(np.float32).transpose(0, 3, 1, 2) / 255.0
    return (x - MEAN) / STD


def load_test_set():
    from torchvision.datasets import CIFAR10
    from common.paths import DATA_DIR
    ds = CIFAR10(DATA_DIR, train=False, download=True)
    return preprocess(ds.data), np.array(ds.targets)


def run(sess, x, bs=250):
    return np.concatenate([sess.run(None, {"input": x[i:i + bs]})[0] for i in range(0, len(x), bs)])


def benchmark(X, Y):
    rows = []
    for name in MODEL_NAMES:
        sess, path = session(name)
        pred = run(sess, X).argmax(1)
        for i in range(30):                      # warm-up
            sess.run(None, {"input": X[i:i + 1]})
        times = []
        for i in range(300):                     # batch-size-1 latency
            t0 = time.perf_counter()
            sess.run(None, {"input": X[i:i + 1]})
            times.append((time.perf_counter() - t0) * 1e3)
        t0 = time.perf_counter()
        run(sess, X)                             # batched throughput
        thr = len(X) / (time.perf_counter() - t0)
        row = {"model": name, "runtime": f"ONNX Runtime {ort.__version__} CPU ({NUM_THREADS} threads)",
               "test_accuracy_%": round(100 * (pred == Y).mean(), 2),
               "onnx_file_MB": round(os.path.getsize(path) / 1e6, 2),
               "latency_bs1_mean_ms": round(float(np.mean(times)), 3),
               "latency_bs1_p95_ms": round(float(np.percentile(times, 95)), 3),
               "throughput_img_per_s": round(thr, 1)}
        rows.append(row)
        print(f"[ONNX-CPU] {name:<20} acc {row['test_accuracy_%']:6.2f}%  size {row['onnx_file_MB']:6.2f} MB  "
              f"latency {row['latency_bs1_mean_ms']:7.3f} ms  throughput {row['throughput_img_per_s']:8.1f} img/s",
              flush=True)
    for r in rows:
        r["latency_speedup_x"] = round(rows[0]["latency_bs1_mean_ms"] / r["latency_bs1_mean_ms"], 2)
        r["throughput_speedup_x"] = round(r["throughput_img_per_s"] / rows[0]["throughput_img_per_s"], 2)
    write_csv(os.path.join(RESULTS_DIR, "onnx_benchmark.csv"), rows)
    print("[ONNX-CPU] results written to results/onnx_benchmark.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODEL_NAMES, help="run single-model inference instead of the benchmark")
    ap.add_argument("--num", type=int, default=10)
    ap.add_argument("--image")
    args = ap.parse_args()

    if args.model is None:
        benchmark(*load_test_set())
        return

    sess, _ = session(args.model)
    if args.image:
        img = np.asarray(Image.open(args.image).convert("RGB").resize((32, 32), Image.BILINEAR))[None]
        x, y = preprocess(img), None
    else:
        X, Y = load_test_set()
        x, y = X[:args.num], Y[:args.num]
    logits = sess.run(None, {"input": x})[0]
    probs = np.exp(logits - logits.max(1, keepdims=True))
    probs /= probs.sum(1, keepdims=True)
    print(f"=== ONNX Runtime CPU inference: resnet18_{args.model}.onnx ===")
    for i, p in enumerate(probs):
        truth = f"  true={CLASSES[y[i]]}" if y is not None else ""
        print(f"  img {i:3d}: pred={CLASSES[p.argmax()]:<10} conf={p.max():.3f}{truth}")


if __name__ == "__main__":
    main()

"""
Reproducible end-to-end execution pipeline (Q1 + Q2 + ONNX deployment).

Runs every step in order with fixed seeds and stores each step's console output in
results/logs/<step>.log.

  python run_pipeline.py                       # everything
  python run_pipeline.py --steps benchmark onnx_export onnx_infer   # selected steps
"""
import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(ROOT, "results", "logs")

STEPS = [
    ("train_baseline", ["Q1_Quantization/train_baseline.py"]),
    ("ptq", ["Q1_Quantization/ptq.py"]),
    ("qat", ["Q1_Quantization/qat.py"]),
    ("unstructured_pruning", ["Q2_Pruning/unstructured_pruning.py"]),
    ("structured_pruning", ["Q2_Pruning/structured_pruning.py"]),
    ("inference", ["inference/inference.py", "--model", "all", "--num", "5"]),
    ("benchmark", ["inference/benchmark.py", "--group", "all"]),
    ("onnx_export", ["onnx_deploy/export_onnx.py"]),
    ("onnx_infer", ["onnx_deploy/onnx_inference.py"]),
]


def run_step(name, cmd):
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"{name}.log")
    print(f"\n{'=' * 70}\n>>> STEP: {name}   ({' '.join(cmd)})\n{'=' * 70}", flush=True)
    t0 = time.time()
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONWARNINGS="ignore")
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen([sys.executable] + cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
        proc.wait()
    if proc.returncode != 0:
        sys.exit(f"step '{name}' failed (exit code {proc.returncode}); see {log_path}")
    print(f">>> {name} finished in {(time.time() - t0) / 60:.1f} min", flush=True)


def main():
    names = [n for n, _ in STEPS]
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", nargs="+", choices=names, default=names)
    args = ap.parse_args()
    for name, cmd in STEPS:
        if name in args.steps:
            run_step(name, cmd)
    print("\nPipeline complete.")


if __name__ == "__main__":
    main()

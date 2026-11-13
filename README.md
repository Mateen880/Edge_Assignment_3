# Assignment 3: Edge Computing & Intelligent Systems (23i-2511_A)

Model optimisation of a ResNet-18 CIFAR-10 classifier:
**Q1** Post-Training Static Quantization (PTQ) and Quantization-Aware Training (QAT),
**Q2** unstructured and structured (channel) pruning, plus ONNX export and ONNX Runtime CPU deployment
of all five models.

## 1. Installation

Python 3.12 was used (Windows 11, CPU only).

```bash
pip install -r requirements.txt
```

`requirements.txt`: torch 2.8.0, torchvision 0.23.0, numpy 2.2.6, pillow, onnx 1.23.2, onnxruntime 1.20.1.
CIFAR-10 (into `data/`) and the ImageNet ResNet-18 weights (torchvision cache) are downloaded automatically
on first run. All scripts use CUDA for training when it is available; quantized INT8 inference and all
benchmarks run on the CPU.

## 2. Project structure

```
common/                     shared code
  models.py                 ResNet-18 (QuantStub/DeQuantStub, FloatFunctional add, fuse_model, prunable mid-channels)
  data.py                   CIFAR-10 loaders, calibration subset, pre-loaded test tensors
  quant_utils.py            backend config, PTQ/QAT prepare, INT8 convert/save/load
  model_zoo.py              load_model(name) for all five models
  utils.py, paths.py        training/eval helpers, sizes, sparsity, CSV writing, paths
Q1_Quantization/
  train_baseline.py         trains the FP32 baseline  -> models/resnet18_fp32.pth
  calibration.py            PTQ calibration pipeline (subset of the training set)
  ptq.py                    fusion, backend, calibration, INT8 conversion -> models/resnet18_ptq_int8.pth
  qat_finetune.py           QAT fine-tuning loop
  qat.py                    QAT init from FP32, fake-quant insertion, fine-tune, convert -> models/resnet18_qat_int8.pth
Q2_Pruning/
  finetune.py               fine-tuning pipeline shared by both pruning methods
  unstructured_pruning.py   iterative global magnitude pruning -> models/resnet18_unstructured_pruned.pth
  structured_pruning.py     L1 channel pruning + architecture reconstruction -> models/resnet18_structured_pruned.pth
inference/
  inference.py              PyTorch inference pipeline for all five models
  benchmark.py              automated benchmark -> results/q1_*.csv, results/q2_*.csv
onnx_deploy/
  export_onnx.py            exports the five models -> models/onnx/*.onnx
  onnx_inference.py         ONNX Runtime CPU inference + benchmark -> results/onnx_benchmark.csv
run_pipeline.py             reproducible end-to-end pipeline (logs in results/logs/)
models/                     trained checkpoints (FP32, PTQ-INT8, QAT-INT8, unstructured, structured) + TorchScript INT8 + ONNX
results/                    CSV results and logs
report/                     technical report (PDF)
```

## 3. Step-by-step execution

Run all commands from the project root.

**Option A: whole pipeline in one command (reproducible, fixed seeds)**
```bash
python run_pipeline.py
```

**Option B: step by step**
```bash
# Q1 / Q2 baseline
python Q1_Quantization/train_baseline.py          # FP32 ResNet-18 (ImageNet init, 6 epochs)
# Q1 quantization
python Q1_Quantization/ptq.py                     # PTQ: fuse -> x86 backend -> calibrate (2000 train imgs) -> INT8
python Q1_Quantization/qat.py                     # QAT: fake-quant fine-tuning (2 epochs) -> INT8
# Q2 pruning
python Q2_Pruning/unstructured_pruning.py         # 80% global sparsity in 4 iterations (+1 epoch fine-tune each)
python Q2_Pruning/structured_pruning.py           # 50% of block-internal channels removed (+3 epochs fine-tune)
# inference and benchmarking (PyTorch, CPU)
python inference/inference.py --model all --num 5
python inference/benchmark.py --group all
# ONNX deployment (ONNX Runtime, CPU)
python onnx_deploy/export_onnx.py
python onnx_deploy/onnx_inference.py
```

Run a single step of the pipeline: `python run_pipeline.py --steps benchmark onnx_export onnx_infer`.

Note on the submitted checkpoints: the baseline run was stopped during epoch 6 (process killed due to low system
memory). `train_baseline.py` saves the best checkpoint after every epoch, so the epoch-5 checkpoint (84.05% test
accuracy) is the FP32 baseline, and the remaining steps were run from it with
`python run_pipeline.py --steps ptq qat unstructured_pruning structured_pruning inference benchmark onnx_export onnx_infer`.

## Results summary (CIFAR-10 test set, CPU, 4 threads)

| Model | Test acc. (%) | Checkpoint (MB) | PyTorch latency bs1 (ms) | ONNX Runtime latency bs1 (ms) |
|---|---|---|---|---|
| FP32 baseline | 84.05 | 44.81 | 10.16 | 2.03 |
| PTQ-INT8 | 83.95 | 11.32 | 6.61 | 1.73 |
| QAT-INT8 | 84.37 | 11.32 | 5.80 | 1.90 |
| Unstructured-pruned (80% sparse) | 84.62 | 44.81 (12.58 zlib) | 8.97 | 2.00 |
| Structured-pruned (5.69 M params) | 82.46 | 22.82 | 6.86 | 0.95 |

Full details are in `results/*.csv` and `report/23i-2511_A_Report.pdf`.

## 4. Inference usage

Model names: `fp32`, `ptq_int8`, `qat_int8`, `unstructured_pruned`, `structured_pruned`.

```bash
python inference/inference.py --model qat_int8 --num 10        # first 10 CIFAR-10 test images
python inference/inference.py --model structured_pruned --eval # full test-set accuracy
python inference/inference.py --model fp32 --image path/to/img.png
python onnx_deploy/onnx_inference.py --model ptq_int8 --num 10 # ONNX Runtime CPU inference
python onnx_deploy/onnx_inference.py --model fp32 --image path/to/img.png
```

## 5. Outputs

| File | Content |
|---|---|
| `results/q1_benchmark.csv` | FP32 / PTQ-INT8 / QAT-INT8: accuracy, size, latency, throughput, speed-ups |
| `results/q1_predictions.csv` | per-image test predictions of the three Q1 models |
| `results/q2_benchmark.csv` | original / unstructured / structured: accuracy, size, params, sparsity, latency, throughput |
| `results/q2_predictions.csv` | per-image test predictions of the three Q2 models |
| `results/onnx_benchmark.csv` | ONNX Runtime CPU results for all five ONNX models |
| `results/logs/*.log` | console output of every pipeline step |

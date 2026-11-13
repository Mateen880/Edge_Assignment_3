"""
PyTorch inference pipeline for all five models:
    fp32 | ptq_int8 | qat_int8 | unstructured_pruned | structured_pruned

Examples
  python inference/inference.py --model ptq_int8 --num 10        # first 10 test images
  python inference/inference.py --model all --num 5
  python inference/inference.py --model qat_int8 --image my_cat.png
  python inference/inference.py --model fp32 --eval              # full test-set accuracy
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from PIL import Image
from torchvision import transforms

from common.data import TEST_TF, load_test_tensors
from common.model_zoo import MODEL_NAMES, load_model
from common.models import CIFAR10_CLASSES


def preprocess_image(path):
    """Any RGB image -> 1x3x32x32 normalised tensor."""
    img = Image.open(path).convert("RGB").resize((32, 32), Image.BILINEAR)
    return TEST_TF(img).unsqueeze(0)


@torch.no_grad()
def predict(model, x):
    probs = torch.softmax(model(x), dim=1)
    conf, cls = probs.max(1)
    return cls, conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all", help=f"one of {MODEL_NAMES} or 'all'")
    ap.add_argument("--image", help="path to an image file to classify")
    ap.add_argument("--num", type=int, default=10, help="number of test images to classify")
    ap.add_argument("--eval", action="store_true", help="evaluate on the full CIFAR-10 test set")
    args = ap.parse_args()
    torch.set_num_threads(4)

    names = MODEL_NAMES if args.model == "all" else (args.model,)
    if args.image:
        x, y = preprocess_image(args.image), None
    else:
        X, Y = load_test_tensors()
        x, y = (X, Y) if args.eval else (X[:args.num], Y[:args.num])

    for name in names:
        model = load_model(name)
        cls, conf = predict(model, x)
        print(f"\n=== {name} ===")
        if args.eval:
            print(f"test accuracy: {100 * (cls == y).float().mean():.2f}%  ({len(y)} images)")
            continue
        for i in range(len(cls)):
            truth = f"  true={CIFAR10_CLASSES[y[i]]:<10}" if y is not None else ""
            print(f"  img {i:3d}: pred={CIFAR10_CLASSES[cls[i]]:<10} conf={conf[i]:.3f}{truth}")


if __name__ == "__main__":
    main()

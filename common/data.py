"""CIFAR-10 data loading (train / test / calibration subset)."""
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from common.paths import DATA_DIR

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

TRAIN_TF = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])
TEST_TF = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])


def _workers():
    return 2


def get_train_loader(batch_size=128, shuffle=True):
    ds = datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=TRAIN_TF)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=_workers(),
                      persistent_workers=True, drop_last=True)


def get_test_loader(batch_size=256):
    ds = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=TEST_TF)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=_workers())


def load_test_tensors():
    """Whole normalised CIFAR-10 test set as two tensors (images, labels).
    Pre-loading removes data-loading time from the inference benchmarks."""
    ds = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=TEST_TF)
    loader = DataLoader(ds, batch_size=1000, shuffle=False, num_workers=0)
    xs, ys = zip(*loader)
    return torch.cat(xs), torch.cat(ys)


def get_calibration_loader(num_images=2000, batch_size=100, seed=0):
    """Random subset of the *training* set (test-time transforms) used for PTQ calibration."""
    ds = datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=TEST_TF)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[:num_images].tolist()
    return DataLoader(Subset(ds, idx), batch_size=batch_size, shuffle=False, num_workers=_workers())

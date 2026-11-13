"""
ResNet-18 implementation used by every part of the assignment.

The network follows the standard ResNet-18 layout (same parameter names as
torchvision, so ImageNet weights can be loaded directly) with three additions:

  * QuantStub / DeQuantStub at the input / output so the same class can be
    converted to an INT8 model with PyTorch eager-mode quantization.
  * The residual addition uses nn.quantized.FloatFunctional so that it can be
    replaced by a quantized add after conversion.
  * `mid_channels` lets each BasicBlock have a reduced number of internal
    channels. This is what the structured channel-pruning code uses to build a
    physically smaller ("reconstructed") architecture.
"""
import torch
import torch.nn as nn
from torch.ao.quantization import QuantStub, DeQuantStub, fuse_modules, fuse_modules_qat

NUM_CLASSES = 10
CIFAR10_CLASSES = ("airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck")

# Number of output channels of each stage and number of blocks per stage
STAGE_CHANNELS = (64, 128, 256, 512)
BLOCKS_PER_STAGE = (2, 2, 2, 2)


class BasicBlock(nn.Module):
    """conv3x3 -> BN -> ReLU -> conv3x3 -> BN  (+ shortcut) -> ReLU"""

    def __init__(self, in_ch, out_ch, stride=1, mid_ch=None):
        super().__init__()
        mid_ch = out_ch if mid_ch is None else mid_ch
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu2 = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch))
        # quantization-friendly residual addition
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.skip_add.add(out, identity)
        return self.relu2(out)


class ResNet18(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, mid_channels=None):
        """
        mid_channels: optional list with one entry per BasicBlock (8 entries)
                      giving the internal channel count of that block.
                      None -> full (unpruned) ResNet-18.
        """
        super().__init__()
        if mid_channels is None:
            mid_channels = [c for c, n in zip(STAGE_CHANNELS, BLOCKS_PER_STAGE) for _ in range(n)]
        self.mid_channels = list(mid_channels)

        self.quant = QuantStub()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        in_ch, idx = 64, 0
        for s, (out_ch, n_blocks) in enumerate(zip(STAGE_CHANNELS, BLOCKS_PER_STAGE)):
            blocks = []
            for b in range(n_blocks):
                stride = 2 if (s > 0 and b == 0) else 1
                blocks.append(BasicBlock(in_ch, out_ch, stride, self.mid_channels[idx]))
                in_ch, idx = out_ch, idx + 1
            setattr(self, f"layer{s + 1}", nn.Sequential(*blocks))

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = torch.flatten(self.avgpool(x), 1)
        x = self.fc(x)
        return self.dequant(x)

    def blocks(self):
        """Iterate over all 8 BasicBlocks in order."""
        for s in range(1, 5):
            yield from getattr(self, f"layer{s}")

    def fuse_model(self, qat=False):
        """Fuse Conv+BN(+ReLU) groups. qat=True keeps the model trainable."""
        fuse = fuse_modules_qat if qat else fuse_modules
        fuse(self, [["conv1", "bn1", "relu"]], inplace=True)
        for blk in self.blocks():
            fuse(blk, [["conv1", "bn1", "relu1"], ["conv2", "bn2"]], inplace=True)
            if blk.downsample is not None:
                fuse(blk.downsample, [["0", "1"]], inplace=True)


def build_resnet18(pretrained_imagenet=False, mid_channels=None):
    """Create the CIFAR-10 ResNet-18. Optionally start from ImageNet weights."""
    model = ResNet18(NUM_CLASSES, mid_channels)
    if pretrained_imagenet:
        from torchvision.models import resnet18, ResNet18_Weights
        src = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).state_dict()
        # the 1000-class ImageNet head is replaced by a new 10-class head
        src = {k: v for k, v in src.items() if not k.startswith("fc.")}
        missing, _ = model.load_state_dict(src, strict=False)
        assert all(k.startswith("fc.") for k in missing), missing
    return model

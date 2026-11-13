"""
QAT fine-tuning loop.

Trains a model that contains FakeQuantize modules so that the weights adapt to INT8
rounding / clamping. During the final epoch the observers are frozen (scales and
zero-points stop changing) and BatchNorm running statistics are frozen, which is the
usual recipe for stable QAT convergence.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.ao.quantization as tq

from common.utils import evaluate, train_one_epoch
from common.quant_utils import convert_to_int8


def qat_finetune(qat_model, train_loader, test_loader, device, epochs=2, lr=1e-3):
    qat_model.to(device)
    optimizer = torch.optim.SGD(qat_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader))

    for epoch in range(1, epochs + 1):
        if epoch == epochs and epochs > 1:
            qat_model.apply(tq.disable_observer)                      # freeze quant params
            qat_model.apply(torch.ao.nn.intrinsic.qat.freeze_bn_stats)  # freeze BN statistics
            print("[QAT] observers and BN statistics frozen for the final epoch")
        t0 = time.time()
        loss, tr_acc = train_one_epoch(qat_model, train_loader, optimizer, device, scheduler, epoch)
        # accuracy of the true INT8 model obtained from the current QAT state
        int8_acc = evaluate(convert_to_int8(qat_model), test_loader, "cpu")
        qat_model.to(device)
        print(f"[QAT] epoch {epoch}/{epochs}  train_loss {loss:.4f}  train_acc {tr_acc:.2f}%  "
              f"INT8 test_acc {int8_acc:.2f}%  ({time.time() - t0:.0f}s)", flush=True)
    return qat_model

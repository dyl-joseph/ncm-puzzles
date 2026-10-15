from contextlib import contextmanager
from unittest.mock import patch

import torch

CHECKPOINT_COPY_METHODS = ("blocking", "batched")


def batched_cpu_tree(value, *, synchronize=None):
    pending = []

    def visit(item):
        if isinstance(item, torch.Tensor):
            source = item.detach()
            if (source.device.type == "mps" and source.layout == torch.strided
                    and source.is_contiguous()):
                # Keep buffers alive if another leaf raises before synchronization.
                pending.append(source)
                destination = source.to(device="cpu", non_blocking=True)
                pending.append(destination)
                return destination
            return source.cpu()
        if isinstance(item, dict):
            return {key: visit(child) for key, child in item.items()}
        if isinstance(item, list):
            return [visit(child) for child in item]
        if isinstance(item, tuple):
            return tuple(visit(child) for child in item)
        return item

    try:
        return visit(value)
    finally:
        if pending:
            (torch.mps.synchronize if synchronize is None else synchronize)()


@contextmanager
def install_batched_cpu_tree(core, *, synchronize=None):
    def cpu_tree(value):
        return batched_cpu_tree(value, synchronize=synchronize)

    with patch.object(core, "cpu_tree", cpu_tree):
        yield

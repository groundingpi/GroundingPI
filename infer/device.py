"""Device selection for the native Transformers service: XPU, CUDA or CPU.

Detection order is XPU, then CUDA, then CPU. An explicit ``--device`` always wins
and is never swapped for another backend. Flash attention is CUDA-only, so any
other device falls back to PyTorch SDPA.
"""
import importlib.util
import logging

import torch

logger = logging.getLogger(__name__)
FLASH = 'flash_attention_2'


def is_available(kind):
    if kind == 'xpu':
        return hasattr(torch, 'xpu') and torch.xpu.is_available()
    if kind == 'cuda':
        return torch.cuda.is_available()
    return kind == 'cpu'


def default_device():
    """Index 0 of the best available accelerator, or ``cpu``."""
    for kind in ('xpu', 'cuda'):
        if is_available(kind):
            return f'{kind}:0'
    return 'cpu'


def resolve_device(device):
    return default_device() if device is None else str(device)


def flash_attn_usable(device):
    return (str(device).split(':', 1)[0] == 'cuda'
            and importlib.util.find_spec('flash_attn') is not None)


def resolve_attn_implementation(requested, device):
    """Keep flash attention only where it can run; otherwise use SDPA.

    ``requested=None`` means the default: flash attention on CUDA when it is
    installed (the previous default), SDPA everywhere else.
    """
    if requested not in (None, FLASH):
        return requested
    if flash_attn_usable(device):
        return FLASH
    if requested == FLASH:
        logger.warning('flash_attention_2 is not usable on %s; using sdpa instead', device)
    return 'sdpa'

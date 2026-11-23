"""Intel XPU entrypoint for the native Transformers reference service.

Runs the unchanged reference server in infer/serve_vlm.py on an Intel GPU. The
device is an explicit XPU and attention is PyTorch SDPA; there is no fallback to
CUDA or the CPU. Existing entrypoints do not import this module.
"""
import argparse
import re
import sys

from infer import serve_vlm

ATTENTION = 'sdpa'
SETUP_HINT = 'See docs/XPU.md for the XPU environment.'


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog='python -m infer.serve_vlm_xpu', description=__doc__, allow_abbrev=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Every other option is passed unchanged to infer/serve_vlm.py '
               '(python -m infer.serve_vlm --help). Attention is fixed to sdpa.')
    parser.add_argument('--device', default='xpu:0', help='Intel GPU to serve on: xpu or xpu:N (default: xpu:0)')
    args, forwarded = parser.parse_known_args(argv)
    if not re.fullmatch(r'xpu(:[0-9]+)?', args.device):
        parser.error(f'--device must be xpu or xpu:N, not {args.device!r}; this entrypoint serves only on Intel GPUs')
    if any(item.startswith('--attn') for item in forwarded):
        parser.error(f'attention is fixed to {ATTENTION} on XPU; remove --attn-implementation')
    if any(item.startswith('--dev') for item in forwarded):
        parser.error('give --device once, as xpu or xpu:N')
    return args.device, forwarded


def require_xpu(device):
    """Stop with an actionable message unless the requested Intel GPU is usable."""
    try:
        import torch
    except ImportError as exc:
        raise SystemExit('PyTorch is not installed in this environment. ' + SETUP_HINT) from exc
    if not (hasattr(torch, 'xpu') and torch.xpu.is_available()):
        raise SystemExit(f'No Intel XPU is available to PyTorch {torch.__version__}. Install a PyTorch XPU build and '
                         'the Intel GPU driver; this entrypoint does not fall back to CUDA or the CPU. ' + SETUP_HINT)
    index, count = int(device.partition(':')[2] or 0), torch.xpu.device_count()
    if index >= count:
        raise SystemExit(f'{device} is not available: PyTorch sees {count} XPU device(s).')
    try:
        torch.empty(1, device=device)  # open the device now, before the weights are loaded
    except RuntimeError as exc:
        raise SystemExit(f'{device} cannot be opened ({exc}); this entrypoint does not fall back to CUDA or '
                         'the CPU. ' + SETUP_HINT) from exc


def main(argv=None):
    device, forwarded = parse_args(sys.argv[1:] if argv is None else argv)
    require_xpu(device)
    # The reference server reads its options from sys.argv; the XPU choices are given explicitly and last.
    sys.argv = [sys.argv[0], *forwarded, '--device', device, '--attn-implementation', ATTENTION]
    serve_vlm.main()


if __name__ == '__main__': main()

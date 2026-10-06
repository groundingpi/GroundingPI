#!/usr/bin/env python3
"""Prepare a downloaded Grounding-EvalData bundle and a full evaluation recipe.

The dataset bundle owns its archive layout and safe extraction implementation.
This entry point verifies those helpers against SHA256SUMS before running them.
It does not download data, start a model service, or overwrite existing recipes.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

try:
    import yaml
except ImportError:
    raise SystemExit('PyYAML is required; install the repository requirements.txt first.')

ROOT = Path(__file__).resolve().parents[1]
HELPERS = ('scripts/prepare_data.py', 'scripts/create_eval_config.py', 'configs/suites.json')


def verify_helpers(bundle):
    """Detect incomplete or modified bundle utilities before executing them."""
    bundle = Path(bundle).expanduser().resolve(strict=True)
    sums = {}
    for line in (bundle / 'SHA256SUMS').read_text(encoding='utf-8').splitlines():
        match = re.fullmatch(r'([0-9a-fA-F]{64}) [ *](.+)', line)
        if not match:
            raise ValueError('Malformed SHA256SUMS entry')
        digest, name = match.groups()
        path = PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name or name in sums:
            raise ValueError('Unsafe or duplicate SHA256SUMS path: ' + name)
        sums[name] = digest.lower()
    for name in HELPERS:
        path = bundle / name
        if name not in sums:
            raise ValueError('Bundle SHA256SUMS does not include ' + name)
        if not path.resolve(strict=True).is_relative_to(bundle):
            raise ValueError('Bundle helper escapes the download directory: ' + name)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != sums[name]:
            raise ValueError('Bundle helper checksum mismatch: ' + name + '; re-download the bundle')
    return bundle


def checkout_file(root, value, label):
    path = Path(value).expanduser()
    path = (path if path.is_absolute() else root / path).resolve()
    if not path.is_relative_to(root) or path.suffix not in ('.yaml', '.yml'):
        raise ValueError(label + ' must be a YAML path inside the checkout')
    return path


def prepare(args, root=ROOT):
    root = Path(root).resolve(strict=True)
    template = checkout_file(root, args.template, 'Template')
    output = checkout_file(root, args.output, 'Output')
    if not template.is_file():
        raise FileNotFoundError('Evaluation template not found: ' + str(template))
    if output.exists():
        raise FileExistsError('Refusing to overwrite existing recipe: ' + str(output))
    config = yaml.safe_load(template.read_text(encoding='utf-8'))
    if not isinstance(config, dict) or config.get('mode') != 'GAM':
        raise ValueError('Data preparation requires a released-model template with mode: GAM; '
                         'keep baseline recipes separate (use gam.yaml for GroundingPI, '
                         'or dlm.yaml, dlm_speculative.yaml, vlm.yaml for GroundAnything)')
    if 'suite' in config and config['suite'] != args.suite:
        raise ValueError('Template suite conflicts with --suite; use the matching suite or a template without suite')
    bundle = verify_helpers(args.bundle)
    data_root = Path(args.data_root).expanduser().resolve()
    # Keep the downloaded bundle intact: annotations are used directly from it.
    if data_root == bundle or data_root.is_relative_to(bundle) or bundle.is_relative_to(data_root):
        raise ValueError('Use separate, non-nested bundle and extracted data directories')
    common = [sys.executable, '-X', 'utf8']
    commands = [
        common + [str(bundle / 'scripts/prepare_data.py'), '--archives-dir', str(bundle),
                  '--data-root', str(data_root), '--code-root', str(root)],
        common + [str(root / 'scripts/check_eval_data.py'), '--bundle', str(bundle),
                  '--data-root', str(data_root)] + (['--deep'] if args.deep_check else []),
        common + [str(bundle / 'scripts/create_eval_config.py'), '--code-root', str(root),
                  '--template', template.relative_to(root).as_posix(), '--data-root', str(data_root),
                  '--suite', args.suite, '--output', output.relative_to(root).as_posix()],
    ]
    for command in commands:
        subprocess.run(command, cwd=root, check=True)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True, type=Path, help='Complete HF download directory')
    parser.add_argument('--data-root', required=True, type=Path, help='Separate extraction directory')
    parser.add_argument('--suite', required=True, choices=('groundingpi34', 'groundanything30'))
    parser.add_argument('--template', required=True, help='Existing evaluation YAML inside this checkout')
    parser.add_argument('--output', required=True, help='New full-suite YAML inside this checkout')
    parser.add_argument('--deep-check', action='store_true', help='Hash and decode all evaluation inputs')
    args = parser.parse_args(argv)
    try:
        prepare(args)
    except (OSError, ValueError, yaml.YAMLError, subprocess.CalledProcessError) as exc:
        parser.exit(2, f'ERROR: {exc}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

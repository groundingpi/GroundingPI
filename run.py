"""Short source-tree commands for setup, training, serving and evaluation."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
ROOT = Path(__file__).resolve().parent

def parser_for(root):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    setup = commands.add_parser('setup', help='install a task environment')
    setup.add_argument('profile', choices=('train', 'serve', 'eval'))
    setup.add_argument('--platform', choices=('gpu', 'ppu'), help='serve platform (default: gpu)')
    train = commands.add_parser('train', help='start training')
    recipe = train.add_mutually_exclusive_group()
    recipe.add_argument('--config', help='project-relative training launch YAML')
    serve = commands.add_parser('serve', help='start the model service')
    engine = serve.add_mutually_exclusive_group()
    engine.add_argument('--config', help='project-relative service launch YAML')
    evaluate = commands.add_parser('eval', help='evaluate an existing model service')
    evaluate.add_argument('--suite', choices=('groundanything30', 'groundingpi34'),
                          help='run a complete named suite, replacing template tasks and sample limit')
    evaluate.add_argument('--run-id', help='new result directory name; --suite otherwise generates a fresh ID')
    evaluation = evaluate.add_mutually_exclusive_group()
    evaluation.add_argument('--config', help='project-relative evaluation YAML')
    actions = [setup, train, serve, evaluate]
    for action in actions:
        action.add_argument('--venv', help='project-relative environment directory (default: .venv-<profile>)')
        action.add_argument('--dry-run', action='store_true', help='print the downstream command without executing it')
    return parser

def relative(root, value):
    path = Path(value)
    target = (root / path).resolve()
    if path.is_absolute() or '..' in path.parts or (not target.is_relative_to(root.resolve())) or (target == root.resolve()):
        raise ValueError('use a path inside the project: ' + value)
    return path.as_posix()

def serve_platform(root, venv):
    record = root / relative(root, venv + '/runtime.json')
    try:
        runtime = json.loads(record.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError('serving environment has no readable platform record; create an environment with python3 run.py setup serve (see environments/README.md), or use --config for a manually prepared environment') from exc
    if not isinstance(runtime, dict) or runtime.get('platform') not in ('gpu', 'ppu'):
        raise ValueError('invalid serving platform in ' + venv + '/runtime.json; expected gpu or ppu')
    return runtime['platform']

def plan(args, root=ROOT):
    if not (root / 'grounding_pi').is_dir() or not (root / 'configs/release').is_dir():
        raise ValueError('run.py must be in a complete GroundingPI source tree')
    profile = args.profile if args.action == 'setup' else {'train': 'train', 'serve': 'serve', 'eval': 'eval'}[args.action]
    venv = relative(root, args.venv or '.venv-' + profile)
    python = str(root / venv / 'bin/python')
    if args.action == 'setup':
        python = sys.executable if sys.version_info[:2] == (3, 12) else shutil.which('python3.12')
        if not python:
            if not args.dry_run:
                raise ValueError('setup requires the interpreter described in environments/README.md (python3.12)')
            python = 'python3.12'
        command = [python, str(root / 'scripts/setup_environment.py'), '--profile', profile, '--venv', venv, '--apply']
        if args.platform and profile != 'serve':
            raise ValueError('--platform applies only to setup serve')
        if profile == 'serve':
            command += ['--platform', args.platform or 'gpu']
    else:
        if args.config:
            name = None
        elif args.action == 'train':
            name = 'vlm_train'
        elif args.action == 'serve':
            name = 'vlm_vllm_' + serve_platform(root, venv)
        else:
            name = 'gam'
        config = relative(root, args.config or f"configs/{('eval' if args.action == 'eval' else 'release')}/{name}.yaml")
        if not (root / config).is_file() or Path(config).suffix not in ('.yaml', '.yml'):
            raise ValueError('configuration YAML not found: ' + config)
        script = 'evaluate.py' if args.action == 'eval' else 'run.py'
        command = [python, str(root / 'scripts' / script), config]
        if args.action == 'eval':
            for flag, value in (('--suite', args.suite), ('--run-id', args.run_id)):
                if value is not None:
                    command += [flag, value]
    if args.action != 'setup' and (not args.dry_run) and (not Path(python).is_file()):
        hint = f'python3 run.py setup {profile} --venv {venv}'
        raise ValueError('environment missing; run: ' + hint)
    return {'profile': profile, 'cwd': str(root), 'command': command}

def main(argv=None):
    parser = parser_for(ROOT)
    args = parser.parse_args(argv)
    try:
        selected = plan(args, ROOT)
    except ValueError as exc:
        parser.error(str(exc))
    if args.dry_run:
        print(json.dumps(selected, indent=2))
        return
    os.chdir(selected['cwd'])
    os.execv(selected['command'][0], selected['command'])
if __name__ == '__main__':
    main()

"""Strict YAML frontend for the existing evaluation protocol and metrics."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
from urllib.parse import urlsplit
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from config_contract import relative
from eval_suites import SUITE_COUNTS, resolve_suite
sys.path.insert(0, str(ROOT))
from eval.utils.eval_data_root import input_path, load_data_paths, task_data_inputs, missing_data_inputs

MODES = ('VLM', 'GAM', 'DLM', 'RLV2', 'REXOMNI', 'LOCATEANYTHING', 'GROUNDINGDINO')
FIELDS = {'datasets', 'mode', 'model_path', 'model_type', 'api_url', 'api_urls', 'data_root', 'output_root',
          'tasks', 'suite', 'run_id', 'limit', 'concurrency', 'coordinate_mode',
          'locate_generation_mode', 'rex_tokenizer_path', 'base_port', 'model_id', 'max_tokens', 'service_contract', 'decoder'}

def task_inventory(root):
    # Read task names without executing !function YAML constructors.
    tasks = {}
    for p in sorted((root / 'eval').rglob('*.yaml')):
        node = yaml.compose(p.read_text())
        names = [v.value for k,v in node.value if k.value == 'task' and isinstance(v, yaml.ScalarNode)] if isinstance(node, yaml.MappingNode) else []
        if names:
            name = names[0]
            if name in tasks:
                raise ValueError(f'duplicate task: {name}')
            tasks[name] = p.relative_to(root).as_posix()
    return tasks

def plan(config, root=ROOT):
    config = resolve_suite(config, root=root)
    if not isinstance(config, dict) or set(config) - FIELDS:
        raise ValueError('unknown evaluation fields or non-mapping configuration')
    required = {'mode', 'model_path', 'api_url', 'data_root', 'output_root', 'tasks', 'run_id'}
    if required - set(config):
        raise ValueError(f'missing fields: {sorted(required - set(config))}')
    model_id = config.get('model_id', config['model_path'])
    if not isinstance(model_id, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_./-]{0,199}', model_id):
        raise ValueError('model_id must be a safe API model identifier')
    maximum = config.get('max_tokens')
    if maximum is not None and (type(maximum) is not int or not 1 <= maximum <= 65536):
        raise ValueError('max_tokens must be an integer in 1..65536')
    service_contract = config.get('service_contract', 'openai')
    if service_contract not in ('openai', 'native'):
        raise ValueError('service_contract must be openai or native')
    mode = config['mode']
    if mode not in MODES:
        raise ValueError(f'mode must be one of {MODES}')
    decoder = config.get('decoder')
    if decoder is not None:
        if decoder not in ('denoise', 'causal', 'speculative'):
            raise ValueError('decoder must be denoise, causal or speculative')
        if mode not in ('DLM', 'RLV2') or service_contract != 'openai':
            raise ValueError('decoder selection requires DLM/RLV2 with the SGLang OpenAI service')
        if not (root / 'infer/decoding.py').is_file():
            raise ValueError('decoder selection requires the optional SGLang decoding adapter in this source tree')
        if config.get('model_type', 'qwen3vl') != 'qwen3vl':
            raise ValueError('SGLang decoder selection requires the VLM/Qwen3 adapter')
        if not config['api_url'].rstrip('/').endswith('/v1'):
            raise ValueError('SGLang API URL must end in /v1 for decoder preflight')
        if decoder == 'denoise' and maximum is not None:
            raise ValueError('denoise uses the DecodeV4 per-task output budgets; omit max_tokens')
    run_id = config['run_id']
    if not isinstance(run_id, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', run_id):
        raise ValueError('run_id must be a safe unique directory name')
    paths = {k: relative(root, config[k]) for k in ('model_path', 'output_root')}
    paths['data_root'] = input_path(root, config['data_root'])
    output = paths['output_root'] / run_id
    relative(root, output.relative_to(root).as_posix())
    # Output must never contain or be inside inputs.
    for k in ('model_path', 'data_root'):
        if output == paths[k] or output in paths[k].parents or paths[k] in output.parents:
            raise ValueError('output overlaps an input directory')
    if ',' in config['model_path'] or '\n' in config['model_path']:
        raise ValueError('model_path must not inject model_args fields')
    api_urls = config.get('api_urls', [config['api_url']])
    if not isinstance(api_urls, list) or not api_urls or any(not isinstance(value, str) for value in api_urls):
        raise ValueError('api_urls must be a nonempty list of URL strings')
    normalized_urls = [value.rstrip('/') for value in api_urls]
    if normalized_urls[0] != config['api_url'].rstrip('/') or len(set(normalized_urls)) != len(normalized_urls):
        raise ValueError('api_urls must start with api_url and contain no duplicates')
    if len(normalized_urls) > 1 and service_contract != 'openai':
        raise ValueError('multiple api_urls require the OpenAI service contract')
    for value in normalized_urls:
        api = urlsplit(value)
        if (',' in value or '\n' in value or api.scheme not in ('http', 'https')
                or not api.hostname or api.username or api.password or api.query or api.fragment):
            raise ValueError('api_urls must be HTTP(S) without embedded credentials/query/fragment')
    tasks = config['tasks']
    known = task_inventory(root)
    if not isinstance(tasks, list) or not tasks or any(not isinstance(t, str) or t not in known for t in tasks) or len(set(tasks)) != len(tasks):
        raise ValueError('tasks must be a nonempty unique list from configs/eval/tasks.json')
    limit = config.get('limit')
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError('limit must be a positive sample count or null for full evaluation')
    concurrency = config.get('concurrency', 4)
    port = config.get('base_port', 12345)
    if type(concurrency) is not int or not 1 <= concurrency <= 256:
        raise ValueError('concurrency must be 1..256 per task')
    if type(port) is not int or not 1024 <= port <= 65535-len(tasks):
        raise ValueError('invalid base_port/task port range')
    coord = config.get('coordinate_mode')
    if mode in ('VLM', 'GROUNDINGDINO'):
        if coord not in ('qwen2', 'qwen3', 'norm01', 'abs'):
            raise ValueError('VLM/GROUNDINGDINO require explicit coordinate_mode; auto is ambiguous')
        if mode == 'GROUNDINGDINO' and coord != 'qwen3':
            raise ValueError('GROUNDINGDINO bridge emits qwen3 0..1000 coordinates')
    elif coord is not None:
        raise ValueError('native-token modes define their own grid; omit coordinate_mode')
    locate = config.get('locate_generation_mode')
    if mode == 'LOCATEANYTHING':
        if locate not in ('fast', 'slow', 'hybrid'):
            raise ValueError('LOCATEANYTHING requires fast/slow/hybrid generation mode')
    elif locate is not None:
        raise ValueError('locate_generation_mode only applies to LOCATEANYTHING')
    rex = config.get('rex_tokenizer_path')
    if mode == 'REXOMNI':
        if not rex:
            raise ValueError('REXOMNI requires its matching tokenizer path')
        paths['rex_tokenizer_path'] = relative(root, rex)
    elif rex is not None:
        raise ValueError('rex_tokenizer_path only applies to REXOMNI')
    model_type = config.get('model_type', 'qwen3vl')
    if model_type not in ('qwen25vl', 'qwen3vl', 'qwen35', 'deepseekvl2'):
        raise ValueError('unsupported legacy request adapter model_type')
    data_paths = load_data_paths(root, config['data_root'], config.get('datasets', 'configs/datasets.yaml'))
    data_inputs = task_data_inputs(root, tasks, known, data_paths)
    for value in data_inputs:
        source = Path(value)
        if source == output or source in output.parents or output in source.parents:
            raise ValueError('output overlaps a dataset input')
    env = {'EVAL_DATA_PATHS': json.dumps(data_paths), 'GAM_EVAL_DATA_ROOT': str(paths['data_root']), 'GAM_EVAL_LOG_ROOT': str(output.relative_to(root)),
           'GAM_EVAL_API_CONCURRENCY_PER_TASK': str(concurrency), 'GAM_EVAL_MODE': mode,
           'HF_HOME': 'outputs/cache/huggingface'}
    if decoder is not None:
        env['GAM_EVAL_DECODER'] = decoder
        if decoder == 'denoise':
            env.update(GAM_DLM_DECODE_PROFILE='task_profiles', GAM_DLM_SINGLE_TARGET_RULE='1')
    env['GAM_EVAL_MODEL_ID'] = model_id
    env['GAM_OPENAI_BASE_URLS'] = ','.join(normalized_urls)
    env['GAM_IMAGE_ROOT'] = data_paths.get('images', str(paths['data_root']))
    env['GAM_MAX_IMAGE_PIXELS'] = str(2560 * 28 * 28)
    env['GAM_METRIC_MODULE'] = 'eval/metrics/detection_metrics.py'
    if maximum is not None: env['GAM_EVAL_MAX_TOKENS_OVERRIDE'] = str(maximum)
    if limit is not None: env['GAM_EVAL_LIMIT'] = str(limit)
    if coord: env['GAM_COORD_MODE'] = coord
    if locate: env['GAM_LOCATEANYTHING_GENERATION_MODE'] = locate
    if rex: env['GAM_REXOMNI_TOKENIZER_PATH'] = str(paths['rex_tokenizer_path'].relative_to(root))
    command = [sys.executable, '-m', 'eval.eval_runner', '--mode', mode,
               '--model_type', model_type, '--model_path', config['model_path'],
               '--tasks', ','.join(tasks), '--api_url', config['api_url'],
               '--job_id', run_id, '--include_path', 'eval', '--skip_cleanup',
               '--base_port', str(port)]
    return {'command': command, 'env': env, 'output': str(output.relative_to(root)),
            'missing_inputs': [str(paths['model_path'])] * (not paths['model_path'].is_dir()) + missing_data_inputs(data_inputs),
            'mode': mode, 'evaluation_scope': 'smoke' if limit else 'full',
            **({'suite': config['suite']} if 'suite' in config else {}),
            'tasks': tasks,
            'effective_runtime': {'decoder': decoder,
                                  'decode_profile': 'DecodeV4' if decoder == 'denoise' else None,
                                  'decode_profile_sha256': hashlib.sha256((root / 'infer/decode/configs/task_profiles.json').read_bytes()).hexdigest() if decoder == 'denoise' else None,
                                  'api_urls': normalized_urls,
                                  'data_root': env['GAM_EVAL_DATA_ROOT'],
                                  'dataset_inputs': data_inputs,
                                  'dataset_paths': data_paths,
                                  'detection_image_root': env['GAM_IMAGE_ROOT'],
                                  'detection_max_image_pixels': int(env['GAM_MAX_IMAGE_PIXELS']),
                                  'detection_metric_module': env['GAM_METRIC_MODULE'],
                                  'detection_metric_sha256': hashlib.sha256((root / env['GAM_METRIC_MODULE']).read_bytes()).hexdigest()},
            'config_sha256': hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()}

def evaluation_environment(prepared, root=ROOT, inherited=None):
    """The YAML entrypoint owns experiment settings; shell credentials remain private."""
    env = dict(os.environ if inherited is None else inherited)
    for key in list(env):
        if key.startswith(('GAM_', 'EVAL_', 'ROBOSPATIAL_', 'SCREENSPOT_', 'LMMS_EVAL_')) or key == 'LM_HARNESS_CACHE_PATH':
            env.pop(key)
    env.update(prepared['env'])
    env['PYTHONPATH'] = str(root / 'vendor/lmms-eval') + os.pathsep + str(root)
    env['GAM_EVAL_VERBOSE_PAYLOAD'] = '0'
    env['GAM_EVAL_RESPONSE_AUDIT_PATH'] = str(root / prepared['output'] / 'responses.jsonl')
    return env


def summarize(output, tasks):
    records = []
    for p in sorted(output.rglob('*.json')):
        if p.name in ('run.json', 'summary.json'): continue
        try: data = json.loads(p.read_text())
        except (ValueError, OSError): continue
        if not isinstance(data, dict) or not isinstance(data.get('results'), dict): continue
        for task, metrics in data['results'].items():
            if task in tasks:
                records.append({'task': task, 'source': p.relative_to(output).as_posix(), 'metrics': metrics})
    return {'records': records, 'missing_tasks': sorted(set(tasks)-{r['task'] for r in records})}

def execute_evaluation(prepared, output, env, root=ROOT):
    """Supervise the runner and leave a final record for cooperative cancellation."""
    from eval.process_control import TerminationRequested, termination_handlers, finish_cleanup, defer_cancellation
    process = None
    code, received_signal, error_type = 1, None, None
    cleanup_timed_out = False
    with termination_handlers():
        try:
            with defer_cancellation():
                process = subprocess.Popen(prepared['command'], cwd=root, env=env, start_new_session=True)
            code = process.wait()
        except (TerminationRequested, KeyboardInterrupt) as exc:
            received_signal = getattr(exc, 'signum', signal.SIGINT)
            code = 128 + received_signal
        except Exception as exc:
            error_type = type(exc).__name__
        finally:
            with finish_cleanup():
                if process is not None and process.poll() is None:
                    try:
                        process.send_signal(received_signal or signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        cleanup_timed_out = True
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass  # Runner may exit between timeout and escalation.
                        process.wait(timeout=5)
                if code < 0:
                    received_signal = -code
                    code = 128 + received_signal
                elif received_signal is None and code in (130, 143):
                    received_signal = code - 128
                summary = summarize(output, prepared['tasks'])
                code = code or (2 if summary['missing_tasks'] else 0)
                status = 'interrupted' if received_signal else ('completed' if code == 0 else 'failed')
                if cleanup_timed_out: status, code = 'cleanup_failed', 1
                summary.update(returncode=code, status=status, termination_signal=received_signal,
                               error_type=error_type, runner_cleanup_timed_out=cleanup_timed_out,
                               mode=prepared['mode'], evaluation_scope=prepared['evaluation_scope'],
                               config_sha256=prepared['config_sha256'])
                temporary = output / '.summary.json.tmp'
                temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
                temporary.replace(output / 'summary.json')
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--suite', choices=tuple(SUITE_COUNTS),
                        help='run the complete named suite, replacing template tasks and sample limit')
    parser.add_argument('--run-id', help='new result directory name; --suite otherwise generates a fresh ID')
    args = parser.parse_args()
    config = resolve_suite(yaml.safe_load(relative(ROOT, args.config).read_text(encoding='utf-8')),
                           root=ROOT, suite=args.suite, run_id=args.run_id)
    prepared = plan(config)
    print(json.dumps(prepared, ensure_ascii=False, indent=2))
    if args.dry_run: return 0
    if prepared['missing_inputs']: raise ValueError('prepare all missing_inputs before execution')
    env = evaluation_environment(prepared)
    # Preflight imports and spawned workers must see the same recipe settings.
    os.environ.clear()
    os.environ.update(env)
    sys.path.insert(0, str(ROOT))
    from models.dependency_contract import verify_dependency
    verify_dependency('lmms_eval', ROOT)
    engine = ROOT / 'vendor/lmms-eval'
    sys.path.insert(0, str(engine))
    for module in ('lmms_eval', 'accelerate'):
        if importlib.util.find_spec(module) is None:
            raise RuntimeError(f'missing {module}; see EVALUATION.md dependency contract')
    from lmms_eval.models import get_model
    get_model('async_openai')  # Fail before creating output if adapter imports are incomplete.
    from eval_preflight import preflight
    service = preflight(config, prepared)
    output = ROOT / prepared['output']
    output.mkdir(parents=True, exist_ok=False)  # Reject reused caches/results.
    (output / 'run.json').write_text(json.dumps({'config':config,'plan':prepared,'service':service},ensure_ascii=False,indent=2)+'\n')
    return execute_evaluation(prepared, output, env)

if __name__ == '__main__':
    raise SystemExit(main())

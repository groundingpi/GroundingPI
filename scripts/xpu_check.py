"""End-to-end and numerical checks for the Intel XPU native service.

  service    Start the XPU service from configs/release/vlm_serve_xpu.yaml (or use
             --base-url for a running one), send the representative cases through
             the HTTP API and the public client, check every answer and the
             request validation, then stop the service. Exit code 1 on a failure.
  reference  Run the same answer cases in process on one device and dtype with the
             reference server's Engine; save greedy answers and first-token logits.
  compare    Compare two reference files, for example XPU bf16 against CPU fp32.

Inputs: docs/assets/demo-poster.jpg and synthetic images drawn here with
Pillow's built-in font, so every expected region is known. Each report keeps
the prompt, the raw output, the parsed result and every check.
"""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grounding_pi import GroundingPi, protocol  # noqa: E402
from grounding_pi.client import parse_response  # noqa: E402

CONFIG = 'configs/release/vlm_serve_xpu.yaml'
MAX_NEW_TOKENS = 8192
POSTER = ROOT / 'docs/assets/demo-poster.jpg'
# Regions on the poster (1920x1080 pixels). The person box is the orange frame drawn in the poster itself.
POSTER_PERSON = (610, 135, 1383, 1075)
POSTER_TITLE = (295, 440, 1610, 680)
POSTER_HEAD = (935, 140, 1150, 400)
CIRCLES = [(60, 80, 260, 280), (560, 380, 720, 540)]
SQUARE = (330, 120, 510, 300)
TRIANGLE = [(620, 100), (740, 300), (500, 300)]
WORDS = [('INTEL', (80, 100)), ('GROUNDING', (80, 300)), ('2026', (520, 460))]
SAVE_BUTTON, CANCEL_BUTTON = (500, 420, 670, 490), (300, 420, 470, 490)


def font(size):
    from PIL import ImageFont
    return ImageFont.load_default(size=size)


def draw_images():
    """Return {name: (PIL image, regions)}; regions are pixel boxes used by the checks."""
    from PIL import Image, ImageDraw
    images = {}
    with Image.open(POSTER) as poster:
        images['poster'] = (poster.convert('RGB'), {})
    shapes = Image.new('RGB', (800, 600), 'white')
    draw = ImageDraw.Draw(shapes)
    for box in CIRCLES: draw.ellipse(box, fill=(220, 30, 30))
    draw.rectangle(SQUARE, fill=(30, 60, 220))
    draw.polygon(TRIANGLE, fill=(30, 170, 60))
    images['shapes'] = (shapes, {})
    text = Image.new('RGB', (800, 600), 'white')
    draw = ImageDraw.Draw(text)
    words = {}
    for word, origin in WORDS:
        draw.text(origin, word, fill='black', font=font(72))
        words[word] = draw.textbbox(origin, word, font=font(72))
    images['text'] = (text, words)
    gui = Image.new('RGB', (800, 600), (235, 235, 235))
    draw = ImageDraw.Draw(gui)
    draw.rectangle((100, 80, 700, 520), fill='white', outline=(120, 120, 120), width=2)
    draw.text((130, 100), 'Save changes?', fill='black', font=font(36))
    draw.rectangle((130, 180, 670, 230), outline=(120, 120, 120), width=2)
    draw.text((140, 190), 'report-final.txt', fill=(80, 80, 80), font=font(26))
    draw.rounded_rectangle(CANCEL_BUTTON, 10, fill=(200, 200, 200))
    draw.text((340, 437), 'Cancel', fill='black', font=font(30))
    draw.rounded_rectangle(SAVE_BUTTON, 10, fill=(30, 100, 220))
    draw.text((550, 437), 'Save', fill='white', font=font(30))
    images['gui'] = (gui, {})
    return images


def to_pixels(coords, size):
    """Map 0..999 grid coordinates to pixels the same way grounding_pi.visualize does."""
    return [v / 999 * (size[0] - 1 if i % 2 == 0 else size[1] - 1) for i, v in enumerate(coords)]


def iou(a, b):
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h
    return w * h / union if union > 0 else 0.0


def inside_box(point, box):
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def inside_polygon(point, polygon):
    x, y, inside = point[0], point[1], False
    for (x1, y1), (x2, y2) in zip(polygon, polygon[1:] + polygon[:1]):
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


# Checks take (predictions, image size, regions) and return (passed, detail).
def boxes_match(label, targets, threshold=0.5):
    def check(predictions, size, regions):
        entries = [p for p in predictions if p['label'] == label]
        if len(entries) != 1 or not entries[0]['coordinates']:
            return False, f'expected one {label!r} entry with boxes'
        boxes = [to_pixels(c, size) for c in entries[0]['coordinates']]
        if len(boxes) != len(targets):
            return False, f'{len(boxes)} boxes for {len(targets)} targets'
        best = [max(iou(b, t) for b in boxes) for t in targets]
        return min(best) >= threshold, f'{label}: IoU ' + ', '.join(f'{v:.3f}' for v in best)
    return check


def all_of(*checks):
    def check(predictions, size, regions):
        results = [c(predictions, size, regions) for c in checks]
        return all(ok for ok, _ in results), '; '.join(detail for _, detail in results)
    return check


def point_in(label, region):
    def check(predictions, size, regions):
        if len(predictions) != 1 or not predictions[0]['coordinates'] or len(predictions[0]['coordinates']) != 1:
            return False, 'expected exactly one point'
        if predictions[0]['label'] != label:
            return False, f'label {predictions[0]["label"]!r} is not {label!r}'
        point = to_pixels(predictions[0]['coordinates'][0], size)
        ok = inside_polygon(point, region) if isinstance(region, list) else inside_box(point, region)
        return ok, 'point at ({:.0f}, {:.0f}) px'.format(*point)
    return check


def ocr_words(predictions, size, regions):
    labels = sorted(p['label'] for p in predictions)
    if labels != sorted(regions) or any(not p['coordinates'] for p in predictions):
        return False, f'transcriptions {labels} differ from {sorted(regions)}'
    scores = {p['label']: max(iou(to_pixels(c, size), regions[p['label']]) for c in p['coordinates'])
              for p in predictions}
    return min(scores.values()) >= 0.5, 'IoU ' + ', '.join(f'{k} {v:.3f}' for k, v in scores.items())


def answered(label):
    def check(predictions, size, regions):
        if len(predictions) != 1 or predictions[0]['label'] != label:
            return False, f'expected one entry for {label!r}'
        if predictions[0]['coordinates'] is None:
            return True, 'answered None'
        return True, 'answered with boxes instead of None (checkpoint behavior, see docs/XPU.md)'
    return check


def case(name, image, task, check, prompt=None, phrase=None):
    """phrase: a referring case sent through the public client; prompt: sent as a raw API request."""
    if phrase is not None:
        prompt = (protocol.build_refer_bbox_prompt if task == 'bbox' else protocol.build_refer_point_prompt)(phrase)
    return {'name': name, 'image': image, 'task': task, 'prompt': prompt, 'client_phrase': phrase, 'check': check}


CASES = [
    case('detection_photo', 'poster', 'bbox', boxes_match('person', [POSTER_PERSON]),
         prompt=protocol.build_dense_bbox_prompt(['person'])),
    case('detection_two_categories', 'shapes', 'bbox',
         all_of(boxes_match('red circle', CIRCLES), boxes_match('blue square', [SQUARE])),
         prompt=protocol.build_dense_bbox_prompt(['red circle', 'blue square'])),
    case('referring_photo', 'poster', 'bbox', boxes_match('the GroundingPI title text', [POSTER_TITLE]),
         phrase='the GroundingPI title text'),
    case('referring_shapes', 'shapes', 'bbox', boxes_match('the blue square', [SQUARE]), phrase='the blue square'),
    case('point_photo', 'poster', 'point', point_in("the robot's head", POSTER_HEAD), phrase="the robot's head"),
    case('point_shapes', 'shapes', 'point', point_in('the green triangle', TRIANGLE), phrase='the green triangle'),
    case('ocr', 'text', 'bbox', ocr_words, prompt=protocol.build_ocr_prompt('text')),
    case('gui_save', 'gui', 'point', point_in('Click the Save button', SAVE_BUTTON),
         prompt=protocol.build_gui_prompt('Click the Save button')),
    case('gui_cancel', 'gui', 'point', point_in('Close the dialog without saving', CANCEL_BUTTON),
         prompt=protocol.build_gui_prompt('Close the dialog without saving')),
    case('no_match', 'shapes', 'bbox', answered('giraffe'), prompt=protocol.build_dense_bbox_prompt(['giraffe'])),
]


def prepare_inputs(output):
    """Write the synthetic images next to the report; return {name: (path, size, regions)}."""
    inputs = {}
    for name, (image, regions) in draw_images().items():
        path = POSTER if name == 'poster' else output / f'{name}.png'
        if name != 'poster': image.save(path)
        inputs[name] = (path, image.size, regions)
    return inputs


def data_uri(path):
    mime = 'image/jpeg' if path.suffix.lower() in ('.jpg', '.jpeg') else 'image/png'
    return f'data:{mime};base64,' + base64.b64encode(path.read_bytes()).decode('ascii')


def request_body(prompt, path, max_tokens=4096):
    """The request grounding_pi.GroundingPi.predict sends (same defaults)."""
    return {'model': 'groundingpi', 'temperature': 0, 'max_tokens': max_tokens,
            'skip_special_tokens': False, 'spaces_between_special_tokens': False,
            'messages': [{'role': 'user', 'content': [
                {'type': 'image_url', 'image_url': {'url': data_uri(path)}}, {'type': 'text', 'text': prompt}]}]}


def evaluate(spec, inputs, response, seconds):
    path, size, regions = inputs[spec['image']]
    result = parse_response(response, spec['task']).to_dict()
    if result['valid']:
        passed, detail = spec['check'](result['predictions'], size, regions)
    else:
        passed, detail = False, 'invalid or incomplete answer: ' + str(result['parse_error'])
    image = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else path.name
    return {'name': spec['name'], 'image': image, 'image_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'prompt': spec['prompt'], 'via': 'client' if spec['client_phrase'] else 'http',
            'raw_output': result['raw_output'], 'predictions': result['predictions'],
            'finish_reason': result['finish_reason'], 'usage': result['usage'],
            'passed': passed, 'check': detail, 'seconds': round(seconds, 3)}


def config_settings():
    import yaml
    config = yaml.safe_load((ROOT / CONFIG).read_text())
    items = config['args']
    options = {items[i][2:]: items[i + 1] for i in range(len(items) - 1)
               if items[i].startswith('--') and not items[i + 1].startswith('--')}
    return options, {k: str(v) for k, v in config.get('env', {}).items()}


def http(method, url, body=None, timeout=900):
    data = None if body is None else json.dumps(body).encode()
    request = Request(url, data=data, method=method, headers={'Content-Type': 'application/json'})
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.status, json.load(response)
    except HTTPError as error:
        return error.code, json.loads(error.read() or b'{}')


def port_in_use(port):
    with socket.socket() as probe:
        return probe.connect_ex(('127.0.0.1', port)) == 0


def wait_ready(root_url, process, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process is not None and process.poll() is not None:
            raise SystemExit(f'the service exited with code {process.returncode}; see service.log')
        try:
            status, health = http('GET', root_url + '/health', timeout=5)
            if status == 200: return health
        except OSError:
            pass
        time.sleep(5)
    raise SystemExit(f'the service was not ready within {timeout} s; see service.log')


def validation_cases(inputs):
    path = inputs['shapes'][0]
    good = request_body(protocol.build_refer_bbox_prompt('the blue square'), path)
    two = json.loads(json.dumps(good))
    two['messages'][0]['content'].insert(0, dict(two['messages'][0]['content'][0]))
    bad_image = json.loads(json.dumps(good))
    bad_image['messages'][0]['content'][0]['image_url']['url'] = 'data:image/png;base64,' + base64.b64encode(b'not an image').decode()
    remote = json.loads(json.dumps(good))
    remote['messages'][0]['content'][0]['image_url']['url'] = 'https://example.com/image.png'
    return [('max_tokens_over_limit', dict(good, max_tokens=MAX_NEW_TOKENS + 1), f'1..{MAX_NEW_TOKENS}'),
            ('unsupported_field', dict(good, logprobs=True), 'unsupported request fields'),
            ('streaming', dict(good, stream=True), 'stream=false'),
            ('unknown_model', dict(good, model='other'), 'unknown model'),
            ('two_images', two, 'too many images'),
            ('invalid_image_data', bad_image, 'invalid image data'),
            ('remote_image_url', remote, 'URLs are not downloaded')]


def service(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    inputs = prepare_inputs(output)
    options, _ = config_settings()
    process, base_url = None, args.base_url
    if base_url is None:
        port = int(options['port'])
        if port_in_use(port): raise SystemExit(f'port {port} is in use; stop that service or pass --base-url')
        command = [sys.executable, 'run.py', 'serve', '--config', CONFIG, '--venv', args.venv]
        log = open(output / 'service.log', 'w')
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        base_url = f'http://{options["host"]}:{port}/v1'
    root_url = base_url[:-3] if base_url.endswith('/v1') else base_url
    report = {'command': None if process is None else ' '.join(command[1:]), 'base_url': base_url}
    try:
        started = time.time()
        report['health'] = wait_ready(root_url, process, args.startup_timeout)
        report['startup_seconds'] = round(time.time() - started, 1)
        status, models = http('GET', base_url + '/models')
        report['models'] = {'status': status, 'ids': [m.get('id') for m in models.get('data', [])]}
        client = GroundingPi(base_url=base_url, model=options['served-model-name'], timeout=args.request_timeout)
        report['cases'] = []
        for spec in CASES:
            path = inputs[spec['image']][0]
            start = time.time()
            if spec['client_phrase'] is not None:
                result = client.predict(path, spec['client_phrase'], task=spec['task'])
                response = {'choices': [{'message': {'content': result.raw_output},
                                         'finish_reason': result.finish_reason}], 'usage': result.usage}
            else:
                status, response = http('POST', base_url + '/chat/completions', request_body(spec['prompt'], path),
                                        timeout=args.request_timeout)
                if status != 200: raise SystemExit(f'{spec["name"]}: HTTP {status} {response}')
            entry = evaluate(spec, inputs, response, time.time() - start)
            report['cases'].append(entry)
            print(('PASS' if entry['passed'] else 'FAIL'), entry['name'], entry['raw_output'], '|', entry['check'], flush=True)
        report['validation'] = []
        for name, body, expected in validation_cases(inputs):
            status, response = http('POST', base_url + '/chat/completions', body)
            message = response.get('error', {}).get('message', '')
            passed = status == 400 and expected in message
            report['validation'].append({'name': name, 'status': status, 'message': message, 'passed': passed})
            print(('PASS' if passed else 'FAIL'), name, status, message, flush=True)
    finally:
        if process is not None:
            os.killpg(process.pid, signal.SIGTERM)
            try: process.wait(60)
            except subprocess.TimeoutExpired: os.killpg(process.pid, signal.SIGKILL)
    checks = ([report['models']['ids'] == [options['served-model-name']]] + [c['passed'] for c in report['cases']]
              + [v['passed'] for v in report['validation']])
    report['passed'], report['total'] = sum(checks), len(checks)
    (output / 'service_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f'{report["passed"]}/{report["total"]} checks passed; report: {output / "service_report.json"}')
    return 0 if all(checks) else 1


def reference(args):
    options, env = config_settings()
    for key, value in env.items(): os.environ.setdefault(key, value)
    import numpy as np
    import torch
    from infer import serve_vlm
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    inputs = prepare_inputs(output)
    engine = serve_vlm.Engine(argparse.Namespace(
        model=options['model'], device=args.device, dtype=args.dtype, attn_implementation='sdpa',
        served_model_name=options['served-model-name'], non_thinking=True,
        max_new_tokens=int(options['max-new-tokens']), max_model_len=int(options['max-model-len']), max_images=1,
        temperature=0.0, top_p=1.0, top_k=0, repetition_penalty=1.0, stop=[]))
    report = {'device': args.device, 'dtype': args.dtype, 'torch': torch.__version__, 'cases': []}
    if args.device.startswith('xpu'):
        report['device_name'] = torch.xpu.get_device_name(torch.device(args.device))
        torch.xpu.reset_peak_memory_stats()
    logits = {}
    for spec in CASES:
        body = request_body(spec['prompt'], inputs[spec['image']][0])
        logits[spec['name']] = first_token_logits(engine, body, serve_vlm, torch).numpy()
        start = time.time()
        response = engine.complete(body)
        report['cases'].append(evaluate(spec, inputs, response, time.time() - start))
        entry = report['cases'][-1]
        print(('PASS' if entry['passed'] else 'FAIL'), entry['name'], entry['raw_output'], '|', entry['check'], flush=True)
    if args.device.startswith('xpu'):
        report['peak_memory_allocated_bytes'] = torch.xpu.max_memory_allocated()
    stem = f'{args.device.split(":")[0]}-{args.dtype}'
    np.savez_compressed(output / f'{stem}-logits.npz', **logits)
    (output / f'{stem}.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f'{sum(c["passed"] for c in report["cases"])}/{len(CASES)} cases passed; saved {output / stem}.json')
    return 0


def first_token_logits(engine, body, serve_vlm, torch):
    """Logits of the first generated token, from the inputs Engine.complete builds for the same request."""
    processor = engine.processor
    messages, images = serve_vlm.read_messages(body['messages'], 1)
    prompt = serve_vlm.render_prompt(processor, messages, True)
    image_inputs = processor.image_processor(images=images, return_tensors='pt')
    merge = processor.image_processor.merge_size ** 2
    prompt = serve_vlm.expand_image_tokens(prompt, [int(grid.prod()) // merge for grid in image_inputs['image_grid_thw']])
    inputs = dict(processor.tokenizer([prompt], return_tensors='pt'), **image_inputs)
    inputs = {k: v.to(engine.args.device) for k, v in inputs.items()}
    with torch.inference_mode():
        return engine.model(**inputs).logits[0, -1].float().cpu()


def compare(args):
    import numpy as np
    runs = []
    for path in (args.candidate, args.reference):
        path = Path(path)
        report = json.loads(path.read_text())
        runs.append((report, np.load(path.with_name(path.stem + '-logits.npz'))))
    (candidate, candidate_logits), (reference, reference_logits) = runs
    rows = []
    for a, b in zip(candidate['cases'], reference['cases']):
        x, y = candidate_logits[a['name']].astype(np.float64), reference_logits[b['name']].astype(np.float64)
        row = {'name': a['name'], 'argmax_equal': int(x.argmax()) == int(y.argmax()),
               'max_abs_diff': float(np.abs(x - y).max()), 'mse': float(((x - y) ** 2).mean()),
               'cosine': float(x @ y / np.linalg.norm(x) / np.linalg.norm(y)),
               'same_answer': a['raw_output'] == b['raw_output'], 'max_coordinate_steps': None,
               'candidate_passed': a['passed'], 'reference_passed': b['passed']}
        pa, pb = a['predictions'], b['predictions']
        if ([(p['label'], [len(c) for c in p['coordinates'] or []]) for p in pa]
                == [(p['label'], [len(c) for c in p['coordinates'] or []]) for p in pb]):
            steps = [abs(u - v) for p, q in zip(pa, pb) for c, d in zip(p['coordinates'] or [], q['coordinates'] or [])
                     for u, v in zip(c, d)]
            row['max_coordinate_steps'] = max(steps, default=0)
        rows.append(row)
    title = f'{candidate["device"]} {candidate["dtype"]} against {reference["device"]} {reference["dtype"]}'
    print(title)
    print(f'{"case":26} argmax  max_abs_diff  mse        cosine    same_answer  coord_steps  passed')
    for r in rows:
        print(f'{r["name"]:26} {str(r["argmax_equal"]):6}  {r["max_abs_diff"]:12.4f}  {r["mse"]:.3e}  {r["cosine"]:.6f}  '
              f'{str(r["same_answer"]):11}  {str(r["max_coordinate_steps"]):11}  {r["candidate_passed"]}/{r["reference_passed"]}')
    if args.output:
        Path(args.output).write_text(json.dumps({'comparison': title, 'cases': rows}, indent=2))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    run = commands.add_parser('service', help='end-to-end check through the XPU service')
    run.add_argument('--venv', default='.venv-xpu', help='project-relative XPU environment (default: .venv-xpu)')
    run.add_argument('--base-url', help='use a running service instead of starting one')
    run.add_argument('--output', default='outputs/xpu_check', help='report directory (default: outputs/xpu_check)')
    run.add_argument('--startup-timeout', type=int, default=1800)
    run.add_argument('--request-timeout', type=int, default=900)
    ref = commands.add_parser('reference', help='run the cases in process on one device and dtype')
    ref.add_argument('--device', required=True, help='for example xpu:0 or cpu')
    ref.add_argument('--dtype', choices=['bfloat16', 'float32'], default='bfloat16')
    ref.add_argument('--output', default='outputs/xpu_check')
    cmp = commands.add_parser('compare', help='compare two reference reports')
    cmp.add_argument('candidate', help='reference report JSON, for example outputs/xpu_check/xpu-bfloat16.json')
    cmp.add_argument('reference', help='reference report JSON, for example outputs/xpu_check/cpu-float32.json')
    cmp.add_argument('--output', help='write the comparison as JSON')
    args = parser.parse_args(argv)
    return {'service': service, 'reference': reference, 'compare': compare}[args.command](args)


if __name__ == '__main__':
    raise SystemExit(main())

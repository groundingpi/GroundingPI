"""Single-device reference chat server using the verified, bundled Transformers fork."""
from infer.device import resolve_attn_implementation, resolve_device
from infer.request_contract import validate_thinking

import argparse
import base64
import importlib
import importlib.util
import io
import json
import math
from pathlib import Path
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from models.dependency_contract import verify_dependency

ROOT = Path(__file__).resolve().parents[1]
IMAGE_TOKEN = '<|image_pad|>'


def bundled_transformers(root=ROOT):
    revision = verify_dependency('transformers', root)
    entry = json.loads((root / 'third_party/manifest.json').read_text())['dependencies']['transformers']
    expected = (root / entry['destination'] / 'src/transformers').resolve()
    module = importlib.import_module('transformers')
    if not Path(module.__file__).resolve().is_relative_to(expected) or module.__version__ != '5.7.0':
        raise ValueError('Inference requires the bundled Transformers 5.7.0 fork. '
                         'Install its prepared source with pip install -e vendor/transformers '
                         'in the Python environment used to launch this server.')
    return module, revision


def render_prompt(processor, messages, non_thinking):
    # The checkpoint owns its assistant prefix. Injecting an empty thinking block
    # changes GroundingPI continuation and can turn grounding into plain text.
    return processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=not non_thinking)


def expand_image_tokens(prompt, counts):
    parts = prompt.split(IMAGE_TOKEN)
    if len(parts) != len(counts) + 1 or any(type(n) is not int or n < 1 for n in counts):
        raise ValueError('image placeholders do not match processed images')
    return parts[0] + ''.join(IMAGE_TOKEN * n + tail for n, tail in zip(counts, parts[1:]))


def truncate_stops(text, stops):
    positions = [text.find(stop) for stop in stops if stop in text]
    return text[:min(positions)] if positions else text


def generation_options(body, limit):
    allowed = {'model', 'messages', 'max_tokens', 'max_completion_tokens', 'temperature',
               'top_p', 'top_k', 'repetition_penalty', 'stop', 'stream', 'n',
               'skip_special_tokens', 'spaces_between_special_tokens', 'chat_template_kwargs'}
    if set(body) - allowed:
        raise ValueError('unsupported request fields: ' + ', '.join(sorted(set(body) - allowed)))
    if body.get('stream', False) is not False or body.get('n', 1) != 1:
        raise ValueError('only stream=false and n=1 are supported')
    if body.get('spaces_between_special_tokens', False) is not False:
        raise ValueError('spaces_between_special_tokens must be false')
    if type(body.get('skip_special_tokens', False)) is not bool:
        raise ValueError('skip_special_tokens must be boolean')
    if 'max_tokens' in body and 'max_completion_tokens' in body:
        raise ValueError('use only one of max_tokens and max_completion_tokens')
    maximum = body.get('max_tokens', body.get('max_completion_tokens', limit))
    if type(maximum) is not int or not 1 <= maximum <= limit:
        raise ValueError(f'max_tokens must be an integer in 1..{limit}')
    values = {key: body.get(key, default) for key, default in
              [('temperature', 0.0), ('top_p', 1.0), ('repetition_penalty', 1.0)]}
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in values.values()):
        raise ValueError('generation parameters must be finite numbers')
    if values['temperature'] < 0 or not 0 < values['top_p'] <= 1 or values['repetition_penalty'] <= 0:
        raise ValueError('invalid temperature, top_p or repetition_penalty')
    top_k = body.get('top_k', 0)
    if type(top_k) is not int or top_k < 0:
        raise ValueError('top_k must be a non-negative integer')
    stops = body.get('stop') or []
    if isinstance(stops, str): stops = [stops]
    if not isinstance(stops, list) or any(not isinstance(s, str) or not s for s in stops):
        raise ValueError('stop must be a non-empty string or list of non-empty strings')
    sample = values['temperature'] > 0
    kwargs = dict(max_new_tokens=maximum, do_sample=sample, repetition_penalty=values['repetition_penalty'])
    if sample: kwargs.update(temperature=values['temperature'], top_p=values['top_p'], top_k=top_k)
    return kwargs, stops


def read_messages(messages, max_images):
    from PIL import Image
    if not isinstance(messages, list) or not messages:
        raise ValueError('messages must be a non-empty list')
    normalized, images = [], []
    for message in messages:
        if not isinstance(message, dict) or message.get('role') not in {'system', 'user', 'assistant'}:
            raise ValueError('only system, user and assistant messages are supported')
        content = message.get('content')
        if isinstance(content, str): content = [{'type': 'text', 'text': content}]
        if not isinstance(content, list): raise ValueError('invalid message content')
        parts = []
        for item in content:
            if not isinstance(item, dict): raise ValueError('invalid content part')
            if item.get('type') == 'text' and isinstance(item.get('text'), str):
                if IMAGE_TOKEN in item['text']: raise ValueError('image placeholders must come from image_url parts')
                parts.append({'type': 'text', 'text': item['text']})
            elif item.get('type') == 'image_url':
                spec = item.get('image_url')
                url = spec.get('url') if isinstance(spec, dict) else None
                if not isinstance(url, str) or not url.startswith('data:image/') or ';base64,' not in url:
                    raise ValueError('image_url requires a base64 image data URI; URLs are not downloaded')
                if len(images) >= max_images: raise ValueError('too many images')
                try:
                    raw = base64.b64decode(url.split(';base64,', 1)[1], validate=True)
                    with Image.open(io.BytesIO(raw)) as image:
                        images.append(image.convert('RGB'))
                except Exception as exc:
                    raise ValueError('invalid image data') from exc
                parts.append({'type': 'image', 'image': 'provided'})
            else:
                raise ValueError('only text and image_url content parts are supported')
        normalized.append({'role': message['role'], 'content': parts})
    if normalized[-1]['role'] != 'user': raise ValueError('the last message must be a user message')
    return normalized, images


class Engine:
    def __init__(self, args):
        # Load the project's path contract explicitly: installed packages may also own "scripts".
        spec = importlib.util.spec_from_file_location('_groundingpi_paths', ROOT / 'scripts/config_contract.py')
        contract = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(contract)
        model_path = contract.relative(ROOT, args.model)
        if not model_path.is_dir(): raise ValueError('model directory is missing')
        transformers, revision = bundled_transformers()
        import torch
        args.device = resolve_device(args.device)
        args.attn_implementation = resolve_attn_implementation(args.attn_implementation, args.device)
        self.torch, self.args = torch, args
        self.defaults = {key: getattr(args, key) for key in
                         ('temperature', 'top_p', 'top_k', 'repetition_penalty', 'stop')}
        generation_options(self.defaults, args.max_new_tokens)
        self.lock = threading.Lock()
        self.processor = transformers.AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True,
                                                                    local_files_only=True)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            str(model_path), trust_remote_code=True, local_files_only=True,
            dtype=getattr(torch, args.dtype), attn_implementation=args.attn_implementation).eval().to(args.device)
        self.info = {'backend': 'bundled-transformers', 'transformers_version': transformers.__version__,
                     'source_revision': revision, 'model': args.served_model_name,
                     'non_thinking': args.non_thinking, 'max_new_tokens': args.max_new_tokens,
                     'max_model_len': args.max_model_len, 'max_images': args.max_images,
                     'service_contract': 'native-v1'}
        print(json.dumps(self.info), flush=True)

    def complete(self, body):
        if not isinstance(body, dict): raise ValueError('request must be a JSON object')
        if body.get('model', self.args.served_model_name) != self.args.served_model_name:
            raise ValueError('unknown model; see /v1/models')
        validate_thinking(body, self.args.non_thinking)
        kwargs, stops = generation_options({**self.defaults, **body}, self.args.max_new_tokens)
        messages, images = read_messages(body.get('messages'), self.args.max_images)
        with self.lock, self.torch.inference_mode():
            prompt = render_prompt(self.processor, messages, self.args.non_thinking)
            image_inputs = self.processor.image_processor(images=images, return_tensors='pt') if images else {}
            merge = self.processor.image_processor.merge_size ** 2
            sizes = [int(grid.prod()) for grid in image_inputs.get('image_grid_thw', [])]
            if any(n % merge for n in sizes): raise ValueError('invalid image grid')
            prompt = expand_image_tokens(prompt, [n // merge for n in sizes])
            inputs = dict(self.processor.tokenizer([prompt], return_tensors='pt'), **image_inputs)
            input_length = inputs['input_ids'].shape[1]
            if input_length + kwargs['max_new_tokens'] > self.args.max_model_len:
                raise ValueError('prompt plus max_tokens exceeds max_model_len')
            inputs = {k: v.to(self.args.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            tokenizer = self.processor.tokenizer
            if stops:
                from transformers import StoppingCriteria, StoppingCriteriaList
                class StopStrings(StoppingCriteria):
                    def __call__(self, input_ids, scores, **unused):
                        text = tokenizer.decode(input_ids[0, input_length:], skip_special_tokens=False)
                        return any(stop in text for stop in stops)
                kwargs['stopping_criteria'] = StoppingCriteriaList([StopStrings()])
            generated = self.model.generate(**inputs, **kwargs)[0, input_length:].tolist()
            eos = self.model.generation_config.eos_token_id
            eos = {eos} if isinstance(eos, int) else set(eos or [])
            trimmed = list(generated)
            while trimmed and trimmed[-1] in eos: trimmed.pop()
            content = tokenizer.decode(trimmed, skip_special_tokens=body.get('skip_special_tokens', False))
            stopped = bool(generated and generated[-1] in eos) or any(stop in content for stop in stops)
            content = truncate_stops(content, stops)
        return {'id': 'chatcmpl-' + uuid.uuid4().hex, 'object': 'chat.completion', 'created': int(time.time()),
                'model': self.args.served_model_name,
                'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': content},
                             'finish_reason': 'stop' if stopped or len(generated) < kwargs['max_new_tokens'] else 'length'}],
                'usage': {'prompt_tokens': input_length, 'completion_tokens': len(generated),
                          'total_tokens': input_length + len(generated)}}


def make_handler(engine, max_request_bytes):
    class Handler(BaseHTTPRequestHandler):
        def send_json(self, status, payload):
            data = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == '/health': self.send_json(200, {'status': 'ok', **engine.info})
            elif self.path == '/v1/models':
                self.send_json(200, {'object': 'list', 'data': [{'id': engine.info['model'], 'object': 'model',
                                                              'owned_by': 'local', 'created': 0}]})
            else: self.send_json(404, {'error': {'message': 'unknown endpoint'}})

        def do_POST(self):
            if self.path != '/v1/chat/completions':
                self.send_json(404, {'error': {'message': 'unknown endpoint'}})
                return
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= max_request_bytes: raise ValueError('invalid or excessive request size')
                body = json.loads(self.rfile.read(size))
                response = engine.complete(body)
            except (ValueError, TypeError) as exc:
                self.send_json(400, {'error': {'message': str(exc), 'type': 'invalid_request_error'}})
                return
            except Exception:
                traceback.print_exc()
                self.send_json(500, {'error': {'message': 'inference failed; inspect server log', 'type': 'server_error'}})
                return
            self.send_json(200, response)
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='project-relative local checkpoint directory')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--served-model-name', default='groundingpi')
    parser.add_argument('--device', default=None, help='default: first XPU, else first CUDA device, else cpu')
    parser.add_argument('--dtype', choices=['bfloat16', 'float16', 'float32'], default='bfloat16')
    parser.add_argument('--attn-implementation', choices=['flash_attention_2', 'sdpa', 'eager'], default=None,
                        help='default: flash_attention_2 on CUDA when installed, else sdpa')
    parser.add_argument('--non-thinking', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top-p', type=float, default=1.0)
    parser.add_argument('--top-k', type=int, default=0)
    parser.add_argument('--repetition-penalty', type=float, default=1.0)
    parser.add_argument('--stop', action='append', default=[])
    parser.add_argument('--max-model-len', type=int, default=4096)
    parser.add_argument('--max-images', type=int, default=1)
    parser.add_argument('--max-request-bytes', type=int, default=32 * 1024 * 1024)
    args = parser.parse_args()
    if min(args.max_new_tokens, args.max_model_len, args.max_images, args.max_request_bytes) < 1:
        parser.error('limits must be positive')
    engine = Engine(args)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(engine, args.max_request_bytes))
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()


if __name__ == '__main__': main()

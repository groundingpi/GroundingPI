# Intel GPU (XPU) Native Service

This XPU integration supports the native Transformers service only. It adds a separate entrypoint, `infer/serve_vlm_xpu.py`, with its own configuration, `configs/release/vlm_serve_xpu.yaml`, and its own environment. The entrypoint runs the unchanged reference server in `infer/serve_vlm.py` with two explicit choices: an XPU device and PyTorch SDPA attention. It stops with an error when no XPU is available and never falls back to CUDA or the CPU. The vLLM launchers, training, evaluation, the client and the existing CUDA and PPU configurations are not changed by it.

## Tested configuration

| Item | Value |
|:---|:---|
| Hardware | Intel Core Ultra 9 285H with its integrated Arc GPU (Arrow Lake) |
| OS and GPU runtime | Ubuntu 24.04, Linux 6.11, i915 driver, Level Zero 1.24 |
| Python / Torch | 3.12 / 2.15.0+xpu from the PyTorch XPU index |
| Transformers | Bundled 5.7.0 fork (`vendor/transformers`, source revision `90dd8674248257e1115e531f2250fd85a32c3863`) |
| Checkpoint | `GroundingPI/GroundingPI`, files verified with its `checksums.sha256` |
| Precision / attention | BF16 / SDPA |
| Image-token budget | `IMAGE_MAX_TOKEN_NUM=512` |
| Limits | 8,192 new tokens per request, 16,384 tokens per sequence, one image per request |
| Decoding | Greedy (temperature 0, top-p 1, top-k 0, repetition penalty 1) |

Other Intel GPUs were not tested.

## Setup

Run from the repository root on Linux x86_64 with Python 3.12 and the Intel GPU driver installed. The environment is new and separate: it does not change `.venv-serve`, `.venv-train` or any other environment.

```bash
python3 -m pip install -r requirements.txt huggingface_hub
hf download GroundingPI/GroundingPI --local-dir weights/vlm

python3 scripts/prepare_dependencies.py --apply --name transformers
python3.12 -m venv .venv-xpu
.venv-xpu/bin/python -m pip install --index-url https://download.pytorch.org/whl/xpu torch==2.15.0
.venv-xpu/bin/python -m pip install -c requirements/serve_xpu.txt -r requirements/serve_xpu.txt -e vendor/transformers
.venv-xpu/bin/python -m pip install --no-deps -e .
.venv-xpu/bin/python -m pip check
.venv-xpu/bin/python -m pip freeze --all > .venv-xpu/installed.freeze.txt
```

`requirements/serve_xpu.txt` pins the tested versions of Torch and of every direct dependency of the bundled Transformers fork. The second install fails if the environment does not hold the XPU build of Torch. Check that PyTorch sees the GPU:

```bash
.venv-xpu/bin/python -c "import torch; print(torch.__version__, torch.xpu.is_available())"
```

## Start the service

```bash
python3 run.py serve --config configs/release/vlm_serve_xpu.yaml --venv .venv-xpu
```

The service listens on `http://127.0.0.1:8000/v1` with model ID `groundingpi`, the same address and API as the other services, so the client example in the README works unchanged, including its default `max_tokens=4096`:

```python
from grounding_pi import GroundingPi

client = GroundingPi()
result = client.predict("your_image.jpg", "the red car", task="bbox")
print(result.to_dict())
```

To select another Intel GPU, change `--device` in the configuration to `xpu:N`. Attention is fixed to SDPA; the entrypoint refuses `--attn-implementation`. All other options are those of `infer/serve_vlm.py`.

## Validate on your hardware

`scripts/xpu_check.py` runs the checkpoint on real hardware. Its `service` command starts the service with the configuration above, sends ten representative requests through the HTTP API and the public client, checks every answer against known regions, checks request validation, and stops the service:

```bash
.venv-xpu/bin/python scripts/xpu_check.py service
```

| Case | Task | Input | Check |
|:---|:---|:---|:---|
| `detection_photo` | Object detection | `docs/assets/demo-poster.jpg`, "person" | IoU of at least 0.5 with the person frame drawn in the poster |
| `detection_two_categories` | Object detection, two categories | Synthetic shapes, "red circle" and "blue square" | Two circles and one square, each with IoU of at least 0.5 |
| `referring_photo`, `referring_shapes` | Referring expression (public client) | Poster title text; the blue square | IoU of at least 0.5 |
| `point_photo`, `point_shapes` | Point localization (public client) | The robot's head; the green triangle | Point inside the region |
| `ocr` | OCR | Synthetic image with three words | Exact transcriptions, IoU of at least 0.5 per word |
| `gui_save`, `gui_cancel` | GUI localization | Synthetic dialog | Point inside the Save or the Cancel button |
| `no_match` | Absent category | Synthetic shapes, "giraffe" | Complete, valid answer for the queried label (see Limitations) |

Request validation: a `max_tokens` above the configured limit, an unsupported field, streaming, an unknown model, two images, invalid image data and a remote image URL must each return HTTP 400 with the matching message. The synthetic images are drawn with Pillow's built-in font and saved next to the report. The report, `outputs/xpu_check/service_report.json`, keeps each input with its SHA-256, the prompt, the raw output, the parsed result, the check and the request time. `service.log` keeps the server output. The command exits with code 1 when any check fails.

To compare devices, run the same cases in process on each device and precision, then compare first-token logits and complete greedy answers:

```bash
.venv-xpu/bin/python scripts/xpu_check.py reference --device xpu:0 --dtype bfloat16
.venv-xpu/bin/python scripts/xpu_check.py reference --device xpu:0 --dtype float32
.venv-xpu/bin/python scripts/xpu_check.py reference --device cpu --dtype float32
.venv-xpu/bin/python scripts/xpu_check.py reference --device cpu --dtype bfloat16
.venv-xpu/bin/python scripts/xpu_check.py compare outputs/xpu_check/xpu-float32.json outputs/xpu_check/cpu-float32.json
.venv-xpu/bin/python scripts/xpu_check.py compare outputs/xpu_check/xpu-bfloat16.json outputs/xpu_check/cpu-float32.json
.venv-xpu/bin/python scripts/xpu_check.py compare outputs/xpu_check/cpu-bfloat16.json outputs/xpu_check/cpu-float32.json
```

XPU FP32 against CPU FP32 isolates the device. CPU BF16 against CPU FP32 shows the error of BF16 itself. XPU BF16 is within tolerance when its first-token argmax matches CPU FP32, its logit error is of the same order as that of CPU BF16, and every case passes its check. Bitwise-identical answers are not expected across devices and precisions: a coordinate token may differ by a few steps of the 0 to 999 grid.

Observed on the tested configuration, over the ten cases:

| Comparison | First-token argmax match | Largest logit difference | Smallest cosine | Identical answers | Largest coordinate difference |
|:---|:---|:---|:---|:---|:---|
| XPU FP32 against CPU FP32 | 10 of 10 | 0.0006 | 1.000000 | 10 of 10 | 0 |
| CPU BF16 against CPU FP32 | 10 of 10 | 2.06 | 0.998863 | 8 of 10 | 14 grid steps (`point_photo`) |
| XPU BF16 against CPU FP32 | 10 of 10 | 1.31 | 0.999029 | 7 of 10 | 14 grid steps (`point_photo`) |

Every case passes its check in all four runs, and the `service` command passes all 18 checks. The two answers that BF16 changes on the CPU (`point_photo`, `point_shapes`) change in the same way on XPU; the third XPU difference is one grid step on one OCR box.

The unit tests in `tests/test_serve_vlm_xpu.py` need neither Torch nor an Intel GPU. The end-to-end test there is optional and runs only on an XPU host:

```bash
python3 -m unittest discover -s tests
GROUNDINGPI_XPU_E2E=1 .venv-xpu/bin/python -m unittest tests.test_serve_vlm_xpu
```

## Limitations

- Only the native Transformers service runs on XPU. vLLM serving, training and FlashAttention are outside this integration.
- One request is processed at a time, as in the reference server.
- The released checkpoint answers absent categories with a box instead of `None` in the tested cases, on CPU FP32 as well as on XPU. The `no_match` case therefore checks for a complete and valid answer, and the report states whether it was `None`.

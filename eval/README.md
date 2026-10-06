<a id="evaluation"></a>

# 🧪 GroundingPI Evaluation

This is the canonical guide for the **34-benchmark suite** used in the GroundingPI paper. Run all commands from the repository root.

[Setup](#setup) · [Data](#data) · [Modes](#modes) · [Full suite](#full-suite) · [Check and run](#run) · [Outputs](#outputs)

<a id="setup"></a>

## 🛠️ Prepare the environment and model service

Use **Linux x86_64 and Python 3.12**. Install the setup dependencies and the separate evaluation environment once:

```bash
python3 -m pip install -r requirements.txt
python3 run.py setup eval
```

Setup installs the bundled evaluation engine and its dependencies. Serving and evaluation use separate environments. Reuse an existing evaluation environment; the installer does not overwrite it. See [Environment Setup](../environments/README.md).

The evaluation profile pins **PyArrow 21.0.0**, which was tested with the released Parquet files. PyArrow 19.0.0 fails on some RefSpatial/RoboSpatial files with `Repetition level histogram size mismatch`. To update an existing evaluation environment, run:

```bash
.venv-eval/bin/python -m pip install "pyarrow==21.0.0"
```

This pin is specific to evaluation; training and serving keep their own dependencies. The data files do not need to be re-encoded.

Download the model and prepare serving through the [repository Quick Start](../README.md#quick-start). Start the matching service in another terminal when ready to evaluate and keep it running throughout the run. Data preparation and checks below do not require model weights, a GPU or a running service.

```bash
python3 run.py serve
```

| Checkpoint | Evaluation template | Model path | API base URL | API model ID |
|:---|:---|:---|:---|:---|
| GroundingPI | `configs/eval/gam.yaml` | `weights/vlm` | `http://127.0.0.1:8000/v1` | `groundingpi` |

The documented serving setup selects the appropriate platform runtime. Evaluation connects to the existing service; it does not start it.

<a id="data"></a>

## 📦 Evaluation data

Download [Grounding-EvalData](https://huggingface.co/datasets/Skywalker0410/Grounding-EvalData), which supplies the evaluation inputs for GroundingPI's **34 entries** and GroundAnything's **30-entry subset**. Install the download utility in your chosen download environment:

```bash
python3 -m pip install -U huggingface_hub
hf download Skywalker0410/Grounding-EvalData \
  --repo-type dataset --local-dir /absolute/path/to/Grounding-EvalData
```

The downloaded directory is the **bundle**: root data archives, `_annotations/`, source notices and verification manifests. Use a separate directory for extracted data. **Keep both directories**: the generated path configuration refers to loose annotations in the bundle as well as extracted images and Parquet files. Moving either directory requires a new local path configuration.

The [preparation command below](#full-suite) verifies the bundle, unpacks it and writes `configs/datasets.local.yaml` plus a full-suite evaluation recipe. It preserves the shipped configuration and source data, reuses identical files, and refuses to overwrite different existing files or configurations. Preserve or rename a conflicting local configuration before generating another one. Original dataset [usage terms](https://huggingface.co/datasets/Skywalker0410/Grounding-EvalData/blob/main/LICENSE) and upstream source conditions remain applicable.

<a id="modes"></a>

## 🧭 Evaluation modes

The evaluator supports **7 mode identifiers**. A mode selects prompts and output parsing independently of the serving engine.

| Mode | Supported models |
|:---|:---|
| **GAM** | **GroundingPI, GroundAnything, GroundAnything-VLM** |
| VLM | Generic vision-language baselines |
| REXOMNI | Rex-Omni |
| LOCATEANYTHING | LocateAnything |
| GROUNDINGDINO | GroundingDINO through a compatible service |
| DLM | Legacy diffusion-checkpoint alias for the GAM protocol |
| RLV2 | Legacy RL-checkpoint alias for the GAM protocol |

**All three released checkpoints use `mode: GAM`.** The `-VLM` suffix does not select the generic `VLM` evaluation mode. GAM uses native spatial tokens and a fixed 0–999 grid; omit `coordinate_mode` for these checkpoints. External baseline weights and services are supplied separately.

<a id="full-suite"></a>

## 📋 Configure the complete 34-benchmark suite

The registry contains **42 task recipes**. These **34 entries** match this paper's suite; selecting every registry entry also includes additional evaluation variants.

| Task group | Entries | Benchmark entries |
|:---|---:|:---|
| Detection | 4 | COCO, LVIS, Dense200, VisDrone |
| Referring boxes | 6 | RefCOCOg val/test; RefCOCO, RefCOCOg, RefCOCO+ family entries; HumanRef |
| Object pointing | 7 | RefCOCOg val/test, COCO, LVIS, Dense200, VisDrone, HumanRef |
| Spatial pointing | 4 | RefSpatial Location/Placement/Unseen; RoboSpatial Context |
| GUI grounding | 3 | ScreenSpot-Pro, ScreenSpot-V2, OSWorld-G |
| OCR | 4 | HierText, ICDAR2015, TotalText, SROIE |
| Document layout | 2 | DocLayNet, M6Doc |
| Visual prompting | 4 | FSC147, Dense200, COCO, LVIS |

The eight excluded recipes are the four `*_Labelless` and four `*_Boxonly` variants.

The RefCOCO family entry `gam_refcocog` is distinct from `gam_refcocog_val` and `gam_refcocog_test`. RefCOCO family and RefSpatial averages in paper tables summarize their component tasks; they are not additional tasks.

The named suite is `groundingpi34`; its exact task list is stored in [configs/eval/suites.json](../configs/eval/suites.json). From the repository root, prepare the downloaded bundle and generate a full evaluation recipe:

```bash
python3 scripts/prepare_eval_data.py \
  --bundle /absolute/path/to/Grounding-EvalData \
  --data-root /absolute/path/to/grounding-eval-data \
  --suite groundingpi34 \
  --template configs/eval/gam.yaml \
  --output configs/eval/full34.local.yaml
```

The helper selects all 34 tasks, sets `mode: GAM` and `limit: null`, assigns a fresh `run_id`, and uses `configs/datasets.local.yaml`. It retains the template's model and service settings. Verify `model_path`, `api_url` and `model_id` against your installation. Evaluation configuration, model and output paths remain relative to the code checkout; data paths may be absolute.

The shipped template remains an **8-sample smoke test**. To select the full suite at runtime, pass `--suite groundingpi34` to `run.py eval`; this uses the named task list, removes the smoke limit and creates a fresh run ID. Retain the task recipes' output-token budgets: changing them changes the evaluation setting.

<a id="data-checks"></a>

### Check the prepared data

Run the data checker independently of model setup. Its basic check validates the required file inventory, sizes and annotation references:

```bash
python3 scripts/check_eval_data.py \
  --bundle /absolute/path/to/Grounding-EvalData \
  --data-root /absolute/path/to/grounding-eval-data
```

For full content hashes and image, Parquet and mask readability, use `--deep` in an environment with Pillow and PyArrow. The evaluation environment includes these dependencies:

```bash
.venv-eval/bin/python scripts/check_eval_data.py \
  --bundle /absolute/path/to/Grounding-EvalData \
  --data-root /absolute/path/to/grounding-eval-data --deep
```

A separate lightweight checking environment only needs `Pillow` and `pyarrow==21.0.0` for deep checks; no model runtime is required. You can also append `--deep-check` to the preparation command when running it with these dependencies installed. Deep validation reads the supplied data without rewriting it, including RefSpatial's external masks and RoboSpatial's `mask_b64`. Data validation does not run model inference or reproduce benchmark scores.

<a id="run"></a>

## ▶️ Check inputs and run

Validate the generated configuration and inspect `missing_inputs` before sending inference requests:

```bash
.venv-eval/bin/python scripts/evaluate.py \
  configs/eval/full34.local.yaml --dry-run
```

This evaluator dry run checks configuration and reports `missing_inputs` without contacting the service. Inspect that field even when the command exits successfully. A missing model-weight directory is a separate setup issue from missing evaluation data; resolve both before inference. The data checker above provides deeper dataset validation. In contrast, `python3 run.py eval --dry-run` only previews the downstream command.

With the GroundingPI service running:

```bash
python3 run.py eval --config configs/eval/full34.local.yaml --suite groundingpi34
```

Actual execution checks the service and required resources before creating the result directory. Keep the service running until evaluation finishes. For a smaller selection, use a separate configuration and run ID and change its `tasks` and sample `limit` explicitly.

<a id="outputs"></a>

## 📊 Inspect the results

Each run writes to `outputs/eval/<run_id>/`:

| Output | Contents |
|:---|:---|
| `run.json` | Selected configuration, launch plan, service checks, and sample scope |
| `responses.jsonl` | Raw predictions, token usage, and finish reasons |
| `log_eval/` and `cache/` | Task logs, engine outputs, and response caches |
| `summary.json` | Per-task metrics, source result files, completion status, and missing tasks |

Check `summary.json` for `status`, `returncode`, and `missing_tasks`. A full-suite run must complete every selected task. Inspect malformed or truncated predictions as well as metric values; a successful HTTP response alone does not establish a correct prediction. Keep the checkpoint revision, selected configuration, and raw results together when comparing runs.

The script reports **per-task results** and does not compute a paper-wide aggregate score. One invocation is **one complete run of the selected suite**; it does not reproduce the paper's ten-run statistical averaging by itself.

<a id="implementation"></a>

## 🗂️ Evaluation implementation

Task definitions live under `Grounding/`, `Dense/`, `Referring/`, `Pointing/`, `GUI/`, `OCR/`, `Layout/`, and `VisualPrompt/`. Shared parsers and data-location helpers are in `utils/`; detection metrics are in `metrics/`.

- [Named suites](../configs/eval/suites.json)
- [Task registry](../configs/eval/tasks.json)
- [Evaluation launcher](../scripts/evaluate.py)
- [Environment setup](../environments/README.md)
- [Return to the project README](../README.md)

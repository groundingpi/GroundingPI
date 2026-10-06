#!/usr/bin/env python3
"""Check prepared Grounding-EvalData inputs on CPU, without loading evaluators.

The default check needs only Python's standard library. --deep additionally
checks every input SHA-256, decodes images/masks, and reads every Parquet batch.
Install its optional readers with: python -m pip install Pillow pyarrow==21.0.0
This validates input data, not model inference or benchmark scores.
"""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import sys


class CheckError(ValueError):
    """An incomplete, inconsistent, unreadable, or unsafe evaluation input."""


def relative_name(value):
    if not isinstance(value, str) or not value or "\\" in value or "\0" in value:
        raise CheckError(f"Unsafe relative path: {value!r}")
    path = PurePosixPath(value)
    if (not path.parts or path.is_absolute() or PureWindowsPath(value).drive
            or any(part == ".." or ":" in part for part in value.split("/"))):
        raise CheckError(f"Unsafe relative path: {value!r}")
    return path.as_posix()


def contained(root, name):
    target = root.joinpath(*relative_name(name).split("/"))
    if not target.resolve().is_relative_to(root):
        raise CheckError(f"Path escapes input root: {name}")
    return target


def integer(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CheckError(f"Expected nonnegative integer for {label}: {value!r}")
    return value


def json_lines(path):
    try:
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except ValueError as exc:
                    raise CheckError(f"Invalid JSON at {path}:{number}: {exc}") from exc
                if not isinstance(row, dict):
                    raise CheckError(f"Expected JSON object at {path}:{number}")
                yield number, row
    except (OSError, UnicodeError) as exc:
        raise CheckError(f"Cannot read UTF-8 JSONL {path}: {exc}") from exc


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise CheckError(f"Cannot read JSON {path}: {exc}") from exc


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def deep_readers():
    try:
        from PIL import Image, ImageFile
        import PIL
        import pyarrow
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise CheckError("--deep needs optional CPU readers. Install them in your evaluation "
                         "environment: python -m pip install Pillow pyarrow==21.0.0") from exc
    # PyArrow 19 can open these footers but fails on full RefSpatial/RoboSpatial
    # reads with 'Repetition level histogram size mismatch'. Version 21 is tested.
    if int(pyarrow.__version__.split(".")[0]) < 21:
        raise CheckError(f"PyArrow {pyarrow.__version__} is too old for the released Parquet "
                         "inputs. Use: python -m pip install pyarrow==21.0.0")
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    return Image, parquet, {"Pillow": PIL.__version__, "pyarrow": pyarrow.__version__}


def decode_image(Image, source, label, png=False):
    try:
        with Image.open(source) as image:
            if png and image.format != "PNG":
                raise CheckError(f"Expected PNG mask: {label}")
            for frame in range(getattr(image, "n_frames", 1)):
                image.seek(frame)
                image.load()
    except Exception as exc:
        raise CheckError(f"Image decode failed for {label}: {exc}") from exc


def check(bundle, data_root, deep=False):
    bundle, data_root = Path(bundle).expanduser().resolve(), Path(data_root).expanduser().resolve()
    roots = {"repository": bundle, "extracted": data_root}
    for name, root in roots.items():
        if not root.is_dir():
            raise CheckError(f"Missing {name} directory: {root}")
    readers = deep_readers() if deep else None
    report = {"passed": False, "deep": deep, "inventory_files": 0, "inventory_bytes": 0,
              "jsonl_files": 0, "jsonl_rows": 0, "json_files": 0,
              "benchmark_tasks_checked": 0, "benchmark_tasks_deferred": [],
              "external_references_checked": 0, "sha256_verified_files": 0,
              "external_images_decoded": 0, "external_masks_decoded": 0,
              "parquet_files": 0, "parquet_rows": 0, "embedded_images_decoded": 0,
              "embedded_masks_decoded": 0, "refspatial_mask_references_checked": 0,
              "model_inference_executed": False}
    inventory, paths = {}, {}
    for line, item in json_lines(bundle / "EVALUATION_FILES.jsonl"):
        name = relative_name(item.get("path"))
        location = item.get("location")
        if not isinstance(location, str) or location not in roots:
            raise CheckError(f"Unknown manifest location at line {line}: {location!r}")
        key = location, name
        if key in inventory:
            raise CheckError(f"Duplicate manifest input: {location}/{name}")
        expected = integer(item.get("bytes"), f"{name} bytes")
        path = contained(roots[location], name)
        if not path.is_file():
            raise CheckError(f"Missing evaluation input: {path}")
        if path.stat().st_size != expected:
            raise CheckError(f"Size mismatch: {path}; expected {expected}, found {path.stat().st_size}")
        inventory[key], paths[key] = item, path
        report["inventory_bytes"] += expected
    if not inventory:
        raise CheckError("EVALUATION_FILES.jsonl contains no inputs")
    report["inventory_files"] = len(inventory)

    references = set()

    def reference(name, label, kind="image"):
        name = relative_name(name)
        key = "extracted", name
        if key not in inventory or inventory[key].get("kind") != kind:
            raise CheckError(f"Referenced {kind} is absent from input manifest: {name} ({label})")
        references.add(key)
        return paths[key]

    def annotation_row(name, row, number):
        if not isinstance(row, dict):
            raise CheckError(f"Expected annotation object: {name}, row {number}")
        if name.startswith("_annotations/"):
            field, prefix = "image_path", "images/"
        elif name.startswith("ScreenSpot-Pro/annotations/"):
            field, prefix = "img_filename", "ScreenSpot-Pro/images/"
        elif name.startswith("ScreenSpot-v2/"):
            field, prefix = "img_filename", "ScreenSpot-v2/screenspotv2_image/"
        elif name == "OSWorld-G/OSWorld-G.json":
            field, prefix = "image_path", "OSWorld-G/images/"
        else:
            raise CheckError(f"Unsupported annotation schema: {name}")
        relative = relative_name(row.get(field))
        reference(prefix + relative, f"{name}, row {number}")

    row_counts = {}
    for key, item in inventory.items():
        name, path = key[1], paths[key]
        if name.endswith(".jsonl"):
            count = 0
            for number, row in json_lines(path):
                if item.get("kind") == "annotation":
                    annotation_row(name, row, number)
                count += 1
            row_counts[key] = count
            report["jsonl_files"] += 1
            report["jsonl_rows"] += count
        elif name.endswith(".json"):
            document = read_json(path)
            report["json_files"] += 1
            if item.get("kind") == "annotation":
                if not isinstance(document, list):
                    raise CheckError(f"Expected annotation JSON array: {name}")
                for number, row in enumerate(document, 1):
                    annotation_row(name, row, number)
                row_counts[key] = len(document)

    if deep:
        print("Checking every evaluation input SHA-256...", file=sys.stderr, flush=True)
        hashes = {}
        for number, row in json_lines(bundle / "validation/evaluation_file_sha256.jsonl"):
            packaged = relative_name(row.get("packaged_relative_path"))
            prefix, separator, name = packaged.partition("/")
            if not separator or prefix not in ("package", "restored"):
                raise CheckError(f"Unknown checksum manifest path at line {number}: {packaged}")
            key = ("repository" if prefix == "package" else "extracted"), relative_name(name)
            if key in hashes:
                raise CheckError(f"Duplicate checksum input: {packaged}")
            expected = row.get("sha256", "")
            if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
                raise CheckError(f"Invalid SHA-256: {packaged}")
            if key not in inventory or integer(row.get("bytes"), packaged) != inventory[key]["bytes"]:
                raise CheckError(f"Checksum manifest disagrees with input inventory: {packaged}")
            hashes[key] = expected.lower()
        if set(hashes) != set(inventory):
            missing = next(iter(set(inventory) - set(hashes)), None)
            raise CheckError(f"Checksum manifest does not cover every evaluation input; missing={missing}")

        def verify_hash(key):
            if sha256(paths[key]) != hashes[key]:
                raise CheckError(f"SHA-256 mismatch: {paths[key]}")
        with ThreadPoolExecutor(max_workers=6) as pool:
            for _ in pool.map(verify_hash, inventory):
                report["sha256_verified_files"] += 1

        Image, parquet, versions = readers
        report["reader_versions"] = versions
        print("Decoding every external image and mask...", file=sys.stderr, flush=True)
        media = [key for key, item in inventory.items() if item.get("kind") in ("image", "mask")]

        def verify_media(key):
            kind = inventory[key]["kind"]
            decode_image(Image, paths[key], str(paths[key]), png=kind == "mask")
            return kind
        with ThreadPoolExecutor(max_workers=6) as pool:
            for kind in pool.map(verify_media, media):
                report["external_images_decoded" if kind == "image" else "external_masks_decoded"] += 1

        def embedded(row, field, label, mask=False):
            value = row.get(field)
            if not isinstance(value, dict) or not isinstance(value.get("bytes"), bytes) or not value["bytes"]:
                raise CheckError(f"Missing embedded {field}.bytes: {label}")
            decode_image(Image, io.BytesIO(value["bytes"]), f"{label}:{field}", png=mask)
            report["embedded_masks_decoded" if mask else "embedded_images_decoded"] += 1

        print("Reading complete Parquet batches and decoding embedded images/masks...", file=sys.stderr, flush=True)
        for key, item in inventory.items():
            name = key[1]
            if not name.endswith(".parquet"):
                continue
            try:
                reader = parquet.ParquetFile(paths[key])
                count = 0
                for batch in reader.iter_batches(batch_size=8):
                    for row in batch.to_pylist():
                        count += 1
                        label = f"{name}, row {count}"
                        if name.startswith(("RefCOCO/", "RefCOCOg/", "RefCOCOplus/")):
                            embedded(row, "image", label)
                        elif name.startswith("RefSpatial-Bench/data/"):
                            embedded(row, "image", label)
                            embedded(row, "mask", label, mask=True)
                            split = {"location": "Location", "placement": "Placement", "unseen": "Unseen"}.get(Path(name).name.split("-")[0])
                            identifier = row.get("id")
                            if split is None or not re.fullmatch(r"[0-9]+", str(identifier)):
                                raise CheckError(f"Invalid RefSpatial split/id: {label}")
                            reference(f"RefSpatial-Bench/{split}/mask/{identifier}.png", label, kind="mask")
                            report["refspatial_mask_references_checked"] += 1
                        elif name.startswith("RoboSpatial-Home/data/"):
                            embedded(row, "img", label)
                            encoded = row.get("mask_b64")
                            if not isinstance(encoded, str) or not encoded:
                                raise CheckError(f"Missing mask_b64: {label}")
                            try:
                                raw = base64.b64decode(encoded, validate=True)
                            except ValueError as exc:
                                raise CheckError(f"Invalid mask_b64: {label}") from exc
                            decode_image(Image, io.BytesIO(raw), f"{label}:mask_b64", png=True)
                            report["embedded_masks_decoded"] += 1
                        else:
                            raise CheckError(f"Unsupported embedded-image Parquet schema: {name}")
                if count != reader.metadata.num_rows:
                    raise CheckError(f"Parquet row count differs from footer: {name}")
                row_counts[key] = count
                report["parquet_rows"] += count
                report["parquet_files"] += 1
            except CheckError:
                raise
            except Exception as exc:
                raise CheckError(f"Parquet read failed: {name}: {exc}. The release was tested with "
                                 "PyArrow 21.0.0; try python -m pip install pyarrow==21.0.0") from exc

    benchmarks = read_json(bundle / "BENCHMARKS.json")
    if not isinstance(benchmarks, list) or not benchmarks:
        raise CheckError("BENCHMARKS.json must contain a nonempty list")
    tasks = set()
    for benchmark in benchmarks:
        if not isinstance(benchmark, dict) or not isinstance(benchmark.get("task_id"), str):
            raise CheckError("Invalid BENCHMARKS.json task entry")
        task = benchmark["task_id"]
        if task in tasks:
            raise CheckError(f"Duplicate benchmark task: {task}")
        tasks.add(task)
        expected = integer(benchmark.get("raw_rows"), f"{task}.raw_rows")
        names = benchmark.get("annotation_files")
        if not isinstance(names, list) or not names:
            raise CheckError(f"Missing annotation_files for {task}")
        keys = []
        for value in names:
            name = relative_name(value)
            candidates = [key for key in (("repository", name), ("extracted", name)) if key in inventory]
            if len(candidates) != 1 or candidates[0] in keys:
                raise CheckError(f"Missing, ambiguous, or repeated benchmark annotation: {task}: {name}")
            keys.append(candidates[0])
        if not deep and any(key[1].endswith(".parquet") for key in keys):
            report["benchmark_tasks_deferred"].append(task)
            continue
        if any(key not in row_counts for key in keys):
            raise CheckError(f"Unsupported benchmark annotation format: {task}")
        actual = sum(row_counts[key] for key in keys)
        if actual != expected:
            raise CheckError(f"Raw row count mismatch: {task}; expected {expected}, found {actual}")
        report["benchmark_tasks_checked"] += 1
    report["external_references_checked"] = len(references)
    report["passed"] = True
    report["scope"] = ("All manifest inputs checked; SHA-256 and full image/Parquet decoding performed."
                       if deep else "All manifest inputs checked for presence/size; JSON/JSONL and their image "
                       "references checked. Parquet row counts, SHA-256 and pixel decoding require --deep.")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True, help="Downloaded Grounding-EvalData repository root")
    parser.add_argument("--data-root", type=Path, required=True, help="Root populated by prepare_data.py")
    parser.add_argument("--deep", action="store_true", help="Verify all hashes and fully decode images and Parquet")
    args = parser.parse_args(argv)
    try:
        report = check(args.bundle, args.data_root, args.deep)
    except (CheckError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print(json.dumps({"passed": False, "error": str(exc), "model_inference_executed": False}, ensure_ascii=False))
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

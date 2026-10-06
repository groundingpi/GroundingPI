"""Small real-format fixtures exercise input failures without model dependencies."""
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/check_eval_data.py"
SPEC = importlib.util.spec_from_file_location("check_eval_data", SCRIPT)
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


def png():
    def chunk(name, data):
        return struct.pack(">I", len(data)) + name + data + struct.pack(">I", zlib.crc32(name + data))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"\0\xff\0\0")) + chunk(b"IEND", b""))


class CheckerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.bundle = Path(self.temp.name) / "bundle"
        self.data = Path(self.temp.name) / "data"
        self.bundle.mkdir(); self.data.mkdir()
        self.items, self.tasks = [], []
        self.annotation = "_annotations/box_eval/COCO.jsonl"
        self.put("repository", self.annotation, "annotation", json.dumps({"image_path": "coco/例子.png"}, ensure_ascii=False).encode("utf-8") + b"\n")
        self.put("extracted", "images/coco/例子.png", "image", png())
        self.tasks.append({"task_id": "gam_coco", "raw_rows": 1, "annotation_files": [self.annotation]})
        self.write_manifests()

    def put(self, location, name, kind, content):
        root = self.bundle if location == "repository" else self.data
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        self.items = [i for i in self.items if (i["location"], i["path"]) != (location, name)]
        self.items.append({"location": location, "path": name, "bytes": len(content), "kind": kind})
        return path

    def write_manifests(self):
        (self.bundle / "EVALUATION_FILES.jsonl").write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in self.items), encoding="utf-8")
        (self.bundle / "BENCHMARKS.json").write_text(json.dumps(self.tasks), encoding="utf-8")
        rows = []
        for i in self.items:
            path = (self.bundle if i["location"] == "repository" else self.data) / i["path"]
            rows.append({"packaged_relative_path": ("package/" if i["location"] == "repository" else "restored/") + i["path"],
                         "bytes": i["bytes"], "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        (self.bundle / "validation").mkdir(exist_ok=True)
        (self.bundle / "validation/evaluation_file_sha256.jsonl").write_text("".join(json.dumps(i) + "\n" for i in rows), encoding="utf-8")

    def run_check(self, deep=False):
        return checker.check(self.bundle, self.data, deep)

    def require_deep(self):
        try:
            checker.deep_readers()
        except checker.CheckError:
            self.skipTest("Optional Pillow/PyArrow 21 readers are not installed")

    def test_utf8_default_and_no_site_packages(self):
        result = self.run_check()
        self.assertTrue(result["passed"])
        self.assertEqual(result["external_references_checked"], 1)
        self.assertEqual(result["benchmark_tasks_checked"], 1)
        environment = dict(os.environ); environment.pop("PYTHONPATH", None)
        process = subprocess.run([sys.executable, "-S", str(SCRIPT), "--bundle", str(self.bundle), "--data-root", str(self.data)], capture_output=True, text=True, encoding="utf-8", env=environment)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertFalse(json.loads(process.stdout)["model_inference_executed"])

    def test_missing_file(self):
        (self.data / "images/coco/例子.png").unlink()
        with self.assertRaisesRegex(checker.CheckError, "Missing evaluation input"):
            self.run_check()

    def test_size_mismatch(self):
        (self.data / "images/coco/例子.png").write_bytes(b"truncated")
        with self.assertRaisesRegex(checker.CheckError, "Size mismatch"):
            self.run_check()

    def test_invalid_jsonl_even_with_updated_size(self):
        self.put("repository", self.annotation, "annotation", b'{"bad":\n')
        self.write_manifests()
        with self.assertRaisesRegex(checker.CheckError, "Invalid JSON"):
            self.run_check()

    def test_raw_row_count_mismatch(self):
        self.tasks[0]["raw_rows"] = 2
        self.write_manifests()
        with self.assertRaisesRegex(checker.CheckError, "Raw row count mismatch"):
            self.run_check()

    def test_missing_external_reference(self):
        self.put("repository", self.annotation, "annotation", b'{"image_path":"coco/not-packaged.png"}\n')
        self.write_manifests()
        with self.assertRaisesRegex(checker.CheckError, "Referenced image is absent"):
            self.run_check()

    def test_unsafe_windows_and_parent_paths(self):
        for value in ["../secret", "C:/secret", "C:secret", "a\\b", "/secret", "image.png:stream", "a/../b"]:
            with self.subTest(value=value), self.assertRaises(checker.CheckError):
                checker.relative_name(value)

    def test_redundant_posix_separators_in_released_references(self):
        self.put("repository", self.annotation, "annotation", json.dumps({"image_path": "coco//例子.png"}, ensure_ascii=False).encode("utf-8") + b"\n")
        self.write_manifests()
        self.assertEqual(self.run_check()["external_references_checked"], 1)

    def test_duplicate_inventory(self):
        manifest = self.bundle / "EVALUATION_FILES.jsonl"
        manifest.write_text(manifest.read_text(encoding="utf-8") + json.dumps(self.items[0]) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(checker.CheckError, "Duplicate manifest"):
            self.run_check()

    def test_invalid_manifest_location_has_clear_diagnostic(self):
        for location in [[], {}, "unknown"]:
            with self.subTest(location=location):
                rows = [dict(i) for i in self.items]
                rows[0]["location"] = location
                (self.bundle / "EVALUATION_FILES.jsonl").write_text("".join(json.dumps(i) + "\n" for i in rows), encoding="utf-8")
                with self.assertRaisesRegex(checker.CheckError, "Unknown manifest location"):
                    self.run_check()

    def test_invalid_gui_json(self):
        name = "ScreenSpot-v2/screenspot_desktop_v2.json"
        self.put("extracted", name, "annotation", b'{not-json}')
        self.write_manifests()
        with self.assertRaisesRegex(checker.CheckError, "Cannot read JSON"):
            self.run_check()

    def test_missing_optional_readers_gives_install_command(self):
        environment = dict(os.environ); environment.pop("PYTHONPATH", None)
        process = subprocess.run([sys.executable, "-S", str(SCRIPT), "--bundle", str(self.bundle), "--data-root", str(self.data), "--deep"], capture_output=True, text=True, encoding="utf-8", env=environment)
        self.assertEqual(process.returncode, 2)
        self.assertIn("pip install Pillow pyarrow==21.0.0", process.stderr)

    def test_same_size_corruption_detected_by_sha(self):
        self.require_deep()
        image = self.data / "images/coco/例子.png"
        content = bytearray(image.read_bytes()); content[-1] ^= 1; image.write_bytes(content)
        self.assertTrue(self.run_check()["passed"])
        with self.assertRaisesRegex(checker.CheckError, "SHA-256 mismatch"):
            self.run_check(True)

    def test_bad_image_with_matching_manifest_hash(self):
        self.require_deep()
        self.put("extracted", "images/coco/例子.png", "image", b"not an image")
        self.write_manifests()
        with self.assertRaisesRegex(checker.CheckError, "Image decode failed"):
            self.run_check(True)

    def test_checksum_inventory_must_cover_all_inputs(self):
        self.require_deep()
        path = self.bundle / "validation/evaluation_file_sha256.jsonl"
        path.write_text(path.read_text(encoding="utf-8").splitlines()[0] + "\n", encoding="utf-8")
        with self.assertRaisesRegex(checker.CheckError, "does not cover every"):
            self.run_check(True)

    def add_parquet(self, name, row):
        self.require_deep()
        import pyarrow as pa
        import pyarrow.parquet as pq
        path = self.data / name
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist([row]), path)
        self.put("extracted", name, "embedded_image_parquet", path.read_bytes())
        self.tasks.append({"task_id": name, "raw_rows": 1, "annotation_files": [name]})
        self.write_manifests()

    def test_full_parquet_embedded_image_and_base64_mask(self):
        self.add_parquet("RoboSpatial-Home/data/context-00000-of-00001.parquet", {"img": {"bytes": png(), "path": "0.png"}, "mask_b64": base64.b64encode(png()).decode()})
        self.assertEqual(len(self.run_check()["benchmark_tasks_deferred"]), 1)
        report = self.run_check(True)
        self.assertEqual(report["parquet_rows"], 1)
        self.assertEqual(report["embedded_images_decoded"], 1)
        self.assertEqual(report["embedded_masks_decoded"], 1)
        self.assertEqual(report["benchmark_tasks_deferred"], [])

    def test_invalid_mask_base64(self):
        self.add_parquet("RoboSpatial-Home/data/context-00000-of-00001.parquet", {"img": {"bytes": png(), "path": "0.png"}, "mask_b64": "not-base64!"})
        with self.assertRaisesRegex(checker.CheckError, "Invalid mask_b64"):
            self.run_check(True)

    def test_refspatial_external_png_mask(self):
        self.put("extracted", "RefSpatial-Bench/Location/mask/0.png", "mask", png())
        self.add_parquet("RefSpatial-Bench/data/location-00000-of-00001.parquet", {"id": 0, "image": {"bytes": png(), "path": "0.png"}, "mask": {"bytes": png(), "path": "0.png"}})
        report = self.run_check(True)
        self.assertEqual(report["refspatial_mask_references_checked"], 1)
        self.assertEqual(report["external_masks_decoded"], 1)


if __name__ == "__main__":
    unittest.main()

"""Device selection for the native service, plus one XPU-versus-CPU parity check.

Everything except the last class runs without an accelerator, so CUDA-only CI stays
green. The parity test compares packed vision-style SDPA attention on XPU (bf16) with the CPU (fp32) and needs an XPU.
"""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("groundingpi_infer_device", ROOT / "infer" / "device.py")
dev = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dev)


def hosts(*, xpu, cuda):
    def fake(kind):
        return {"xpu": xpu, "cuda": cuda}.get(kind, kind == "cpu")
    return mock.patch.object(dev, "is_available", side_effect=fake)


class DetectionTest(unittest.TestCase):
    def test_xpu_preferred_over_cuda(self):
        with hosts(xpu=True, cuda=True):
            self.assertEqual(dev.default_device(), "xpu:0")

    def test_cuda_when_no_xpu(self):
        with hosts(xpu=False, cuda=True):
            self.assertEqual(dev.default_device(), "cuda:0")

    def test_cpu_without_accelerator(self):
        with hosts(xpu=False, cuda=False):
            self.assertEqual(dev.default_device(), "cpu")

    def test_explicit_device_wins(self):
        with hosts(xpu=True, cuda=True):
            self.assertEqual(dev.resolve_device("cuda:1"), "cuda:1")
            self.assertEqual(dev.resolve_device(torch.device("xpu:1")), "xpu:1")
            self.assertEqual(dev.resolve_device(None), "xpu:0")


class AttentionTest(unittest.TestCase):
    def test_flash_kept_on_cuda_when_installed(self):
        with mock.patch.object(importlib.util, "find_spec", return_value=object()):
            self.assertEqual(dev.resolve_attn_implementation(None, "cuda:0"), "flash_attention_2")
            self.assertEqual(dev.resolve_attn_implementation("flash_attention_2", "cuda:0"), "flash_attention_2")

    def test_flash_missing_on_cuda_falls_back(self):
        with mock.patch.object(importlib.util, "find_spec", return_value=None):
            self.assertEqual(dev.resolve_attn_implementation(None, "cuda:0"), "sdpa")

    def test_flash_never_used_off_cuda(self):
        with mock.patch.object(importlib.util, "find_spec", return_value=object()):
            for device in ("xpu:0", "cpu"):
                self.assertEqual(dev.resolve_attn_implementation(None, device), "sdpa")
                with self.assertLogs(dev.logger, "WARNING"):
                    self.assertEqual(dev.resolve_attn_implementation("flash_attention_2", device), "sdpa")

    def test_explicit_non_flash_choice_is_kept(self):
        self.assertEqual(dev.resolve_attn_implementation("eager", "xpu:0"), "eager")
        self.assertEqual(dev.resolve_attn_implementation("sdpa", "cuda:0"), "sdpa")


class ServeDefaultsTest(unittest.TestCase):
    def test_serve_cli_defaults_to_detection(self):
        source = (ROOT / "infer" / "serve_vlm.py").read_text()
        self.assertNotIn("default='cuda:0'", source)
        self.assertIn("resolve_device(args.device)", source)


@unittest.skipUnless(hasattr(torch, "xpu") and torch.xpu.is_available(), "needs an Intel XPU")
class XpuParityTest(unittest.TestCase):
    def test_sdpa_vision_attention_matches_cpu(self):
        torch.manual_seed(0)
        lengths = [5, 9]
        q, k, v = (torch.randn(sum(lengths), 4, 16) for _ in range(3))
        cu = torch.tensor([0, 5, 14])

        def packed(device, dtype):
            outs = []
            for a, b in zip(cu[:-1].tolist(), cu[1:].tolist()):
                parts = [t[a:b].to(device, dtype).transpose(0, 1) for t in (q, k, v)]
                outs.append(torch.nn.functional.scaled_dot_product_attention(*parts).transpose(0, 1))
            return torch.cat(outs).float().cpu()

        torch.testing.assert_close(packed("xpu", torch.bfloat16), packed("cpu", torch.float32), atol=3e-2, rtol=3e-2)


if __name__ == "__main__":
    unittest.main()

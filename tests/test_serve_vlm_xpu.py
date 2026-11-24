"""Tests for the Intel XPU entrypoint. They need neither Torch nor an Intel GPU, except
XpuServiceTest, which runs only when GROUNDINGPI_XPU_E2E=1 is set on an XPU host."""
from contextlib import redirect_stderr
import io
import os
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from infer import serve_vlm, serve_vlm_xpu  # noqa: E402


def fake_torch(available=True, count=1, opens=True):
    def empty(*shape, device):
        if not opens: raise RuntimeError('No XPU devices are available.')
    xpu = types.SimpleNamespace(is_available=lambda: available, device_count=lambda: count)
    return types.SimpleNamespace(__version__='0.0+test', xpu=xpu, empty=empty)


class XpuEntrypointTest(unittest.TestCase):
    def run_main(self, argv, torch=None):
        torch = torch or fake_torch()
        with mock.patch.dict(sys.modules, {'torch': torch}), \
                mock.patch.object(serve_vlm_xpu.serve_vlm, 'main') as server, \
                mock.patch.object(sys, 'argv', ['serve_vlm_xpu']):
            serve_vlm_xpu.main(argv)
            return server, list(sys.argv)

    def test_forwards_explicit_xpu_device_and_sdpa(self):
        server, argv = self.run_main(['--model', 'weights/vlm', '--max-new-tokens', '8192'])
        server.assert_called_once_with()
        self.assertEqual(argv[1:], ['--model', 'weights/vlm', '--max-new-tokens', '8192',
                                    '--device', 'xpu:0', '--attn-implementation', 'sdpa'])

    def test_selects_requested_xpu(self):
        _, argv = self.run_main(['--model', 'weights/vlm', '--device', 'xpu:1'], fake_torch(count=2))
        self.assertEqual(argv[-4:], ['--device', 'xpu:1', '--attn-implementation', 'sdpa'])

    def test_rejects_other_devices(self):
        for device in ('cuda:0', 'cpu', 'cuda', 'xpu:a'):
            with self.subTest(device=device), self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
                serve_vlm_xpu.parse_args(['--device', device])

    def test_rejects_attention_override(self):
        for argv in (['--attn-implementation', 'flash_attention_2'], ['--attn-implementation=sdpa'], ['--attn', 'eager']):
            with self.subTest(argv=argv), self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
                serve_vlm_xpu.parse_args(argv)

    def test_does_not_fall_back_without_xpu(self):
        with self.assertRaises(SystemExit) as raised:
            self.run_main(['--model', 'weights/vlm'], fake_torch(available=False))
        self.assertIn('does not fall back to CUDA or the CPU', str(raised.exception))

    def test_missing_device_index(self):
        with self.assertRaises(SystemExit) as raised:
            self.run_main(['--device', 'xpu:1'], fake_torch(count=1))
        self.assertIn('xpu:1 is not available', str(raised.exception))

    def test_stops_before_loading_when_device_cannot_open(self):
        with self.assertRaises(SystemExit) as raised:
            self.run_main(['--model', 'weights/vlm'], fake_torch(opens=False))
        self.assertIn('xpu:0 cannot be opened', str(raised.exception))

    def test_reference_server_defaults_are_unchanged(self):
        with mock.patch.object(sys, 'argv', ['serve_vlm', '--model', 'weights/vlm']), \
                mock.patch.object(serve_vlm, 'Engine', side_effect=RuntimeError('stop')) as engine:
            with self.assertRaises(RuntimeError):
                serve_vlm.main()
        args = engine.call_args.args[0]
        self.assertEqual((args.device, args.attn_implementation), ('cuda:0', 'flash_attention_2'))

    def test_help_works_without_site_packages(self):
        result = subprocess.run([sys.executable, '-S', '-m', 'infer.serve_vlm_xpu', '--help'], cwd=ROOT,
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('xpu:N', result.stdout)


@unittest.skipUnless(os.environ.get('GROUNDINGPI_XPU_E2E') == '1',
                     'set GROUNDINGPI_XPU_E2E=1 on an Intel XPU host with weights/vlm and .venv-xpu')
class XpuServiceTest(unittest.TestCase):
    def test_end_to_end(self):
        result = subprocess.run([sys.executable, 'scripts/xpu_check.py', 'service'], cwd=ROOT, timeout=3600)
        self.assertEqual(result.returncode, 0, 'see outputs/xpu_check/service_report.json')


if __name__ == '__main__':
    unittest.main()

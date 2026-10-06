import argparse
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('prepare_eval_data', Path(__file__).resolve().parents[1] / 'scripts/prepare_eval_data.py')
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)
        self.bundle, self.root = base / 'download', base / 'checkout'
        self.bundle.mkdir()
        self.root.mkdir()
        lines = []
        for name in prepare.HELPERS:
            path = self.bundle / name
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(b'# fixture\n')
            lines.append(hashlib.sha256(path.read_bytes()).hexdigest() + '  ' + name)
        (self.bundle / 'SHA256SUMS').write_text('\n'.join(lines), encoding='utf-8')
        (self.root / 'template.yaml').write_text('mode: GAM\n', encoding='utf-8')
        self.args = argparse.Namespace(bundle=self.bundle, data_root=base / 'extracted',
                                      template='template.yaml', output='full.yaml',
                                      suite='groundingpi34', deep_check=True)

    def test_modified_helper_rejected_before_execution(self):
        (self.bundle / prepare.HELPERS[0]).write_bytes(b'changed')
        with patch.object(prepare.subprocess, 'run') as run:
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                prepare.prepare(self.args, self.root)
            run.assert_not_called()

    def test_existing_recipe_preserved(self):
        output = self.root / 'full.yaml'
        output.write_bytes(b'custom settings')
        with self.assertRaises(FileExistsError):
            prepare.prepare(self.args, self.root)
        self.assertEqual(output.read_bytes(), b'custom settings')

    def test_output_cannot_escape_checkout(self):
        self.args.output = '../outside.yaml'
        with self.assertRaisesRegex(ValueError, 'inside the checkout'):
            prepare.prepare(self.args, self.root)

    def test_baseline_template_rejected_without_changing_its_mode(self):
        template = self.root / 'template.yaml'
        original = b'mode: VLM\ncoordinate_mode: qwen3\n'
        template.write_bytes(original)
        with patch.object(prepare.subprocess, 'run') as run:
            with self.assertRaisesRegex(ValueError, 'mode: GAM'):
                prepare.prepare(self.args, self.root)
            run.assert_not_called()
        self.assertEqual(template.read_bytes(), original)

    def test_dataset_must_not_be_extracted_over_bundle(self):
        self.args.data_root = self.bundle
        with self.assertRaisesRegex(ValueError, 'non-nested'):
            prepare.prepare(self.args, self.root)

    def test_conflicting_template_suite_rejected_before_extraction(self):
        (self.root / 'template.yaml').write_text('mode: GAM\nsuite: groundanything30\n', encoding='utf-8')
        with patch.object(prepare.subprocess, 'run') as run:
            with self.assertRaisesRegex(ValueError, 'suite conflicts'):
                prepare.prepare(self.args, self.root)
            run.assert_not_called()

    def test_failed_data_check_does_not_generate_recipe(self):
        failure = subprocess.CalledProcessError(2, ['check_eval_data.py'])
        with patch.object(prepare.subprocess, 'run', side_effect=[None, failure]) as run:
            with self.assertRaises(subprocess.CalledProcessError):
                prepare.prepare(self.args, self.root)
        self.assertEqual(run.call_count, 2)
        self.assertIn('--deep', run.call_args.args[0])
        self.assertFalse((self.root / 'full.yaml').exists())

    def test_success_uses_same_interpreter_and_selected_suite(self):
        with patch.object(prepare.subprocess, 'run') as run:
            prepare.prepare(self.args, self.root)
        self.assertEqual(run.call_count, 3)
        for call in run.call_args_list:
            self.assertEqual(call.args[0][:3], [prepare.sys.executable, '-X', 'utf8'])
            self.assertTrue(call.kwargs['check'])
        command = run.call_args.args[0]
        self.assertEqual(command[command.index('--suite') + 1], 'groundingpi34')


if __name__ == '__main__':
    unittest.main()

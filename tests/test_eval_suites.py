"""CPU tests for full-suite selection and source-tree CLI forwarding."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from eval_suites import SUITE_COUNTS, resolve_suite, suite_tasks
from evaluate import evaluation_environment, plan as evaluation_plan

spec = importlib.util.spec_from_file_location('suite_source_tree_cli', ROOT / 'run.py')
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)
IS_GROUNDANYTHING = (ROOT / 'grounding_anything').is_dir()
DEFAULT_SUITE = 'groundanything30' if IS_GROUNDANYTHING else 'groundingpi34'
DEFAULT_TEMPLATE = 'dlm.yaml' if IS_GROUNDANYTHING else 'gam.yaml'


def template(name=DEFAULT_TEMPLATE):
    return yaml.safe_load((ROOT / 'configs/eval' / name).read_text(encoding='utf-8'))


class SuiteResolutionTests(unittest.TestCase):
    def test_published_suite_counts_order_and_coverage(self):
        thirty = suite_tasks('groundanything30', ROOT)
        thirty_four = suite_tasks('groundingpi34', ROOT)
        self.assertEqual(len(thirty), 30)
        self.assertEqual(len(thirty_four), 34)
        self.assertEqual(thirty_four[:30], thirty)
        self.assertEqual(thirty[:4], ['gam_coco', 'gam_lvis', 'gam_dense200', 'gam_visdrone'])
        self.assertEqual(thirty_four[30:], ['gam_humanref', 'gam_rex_point_humanref', 'gam_visual_coco', 'gam_visual_lvis'])
        self.assertFalse(any(name.endswith(('_Labelless', '_Boxonly')) for name in thirty_four))
        thirty.clear()
        self.assertEqual(len(suite_tasks('groundanything30', ROOT)), 30)

    def test_cli_suite_replaces_real_smoke_scope_and_preserves_model_settings(self):
        names = ['dlm.yaml', 'dlm_speculative.yaml', 'vlm.yaml'] if IS_GROUNDANYTHING else ['gam.yaml']
        for name in names:
            with self.subTest(template=name):
                original = template(name)
                before = deepcopy(original)
                self.assertEqual(original['limit'], 8)
                selected = resolve_suite(original, root=ROOT, suite=DEFAULT_SUITE)
                self.assertEqual(original, before)
                self.assertEqual(selected['tasks'], suite_tasks(DEFAULT_SUITE, ROOT))
                self.assertIsNone(selected['limit'])
                self.assertEqual(selected['suite'], DEFAULT_SUITE)
                for field in set(original) - {'tasks', 'limit', 'run_id'}:
                    self.assertEqual(selected[field], original[field], field)
                other = resolve_suite(original, root=ROOT, suite=DEFAULT_SUITE)
                self.assertNotEqual(selected['run_id'], other['run_id'])
                self.assertTrue(selected['run_id'].startswith(DEFAULT_SUITE + '_'))
                self.assertLessEqual(len(selected['run_id']), 80)

    def test_explicit_run_id_and_cli_suite_override_yaml_selection(self):
        original = {**template(), 'suite': 'groundanything30', 'limit': 3}
        selected = resolve_suite(original, root=ROOT, suite='groundingpi34', run_id='comparison_01')
        self.assertEqual(selected['run_id'], 'comparison_01')
        self.assertEqual(selected['tasks'], suite_tasks('groundingpi34', ROOT))
        self.assertIsNone(selected['limit'])

    def test_yaml_suite_expands_without_changing_explicit_run_id(self):
        config = template()
        config.pop('tasks')
        config.pop('limit')
        config['suite'] = DEFAULT_SUITE
        selected = resolve_suite(config, root=ROOT)
        self.assertEqual(selected['tasks'], suite_tasks(DEFAULT_SUITE, ROOT))
        self.assertIsNone(selected['limit'])
        self.assertEqual(selected['run_id'], config['run_id'])
        self.assertEqual(resolve_suite(selected, root=ROOT), selected)

    def test_yaml_suite_rejects_finite_limits_or_conflicting_task_lists(self):
        for limit in (8, 1, 0, False):
            with self.subTest(limit=limit), self.assertRaisesRegex(ValueError, 'full evaluation'):
                resolve_suite({'suite': DEFAULT_SUITE, 'limit': limit}, root=ROOT)
        tasks = suite_tasks(DEFAULT_SUITE, ROOT)
        for conflicting in (tasks[:-1], list(reversed(tasks)), [], 'gam_coco'):
            with self.subTest(tasks=conflicting), self.assertRaisesRegex(ValueError, 'tasks conflict'):
                resolve_suite({'suite': DEFAULT_SUITE, 'limit': None, 'tasks': conflicting}, root=ROOT)

    def test_unknown_suite_and_invalid_run_id_are_rejected(self):
        for name in ('all', '', None, ['groundingpi34']):
            with self.subTest(suite=name), self.assertRaises(ValueError):
                resolve_suite({'suite': name}, root=ROOT)
        for run_id in ('../old', '', 'bad name', 'a' * 81):
            with self.subTest(run_id=run_id), self.assertRaises(ValueError):
                resolve_suite(template(), root=ROOT, suite=DEFAULT_SUITE, run_id=run_id)

    def test_malformed_suite_inventory_and_missing_registry_task_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = root / 'configs/eval'
            directory.mkdir(parents=True)
            correct = suite_tasks(DEFAULT_SUITE, ROOT)
            registry = {name: 'unused.yaml' for name in correct}
            (directory / 'tasks.json').write_text(json.dumps(registry), encoding='utf-8')
            for malformed in (correct[:-1], correct[:-1] + [correct[0]], correct[:-1] + [None]):
                (directory / 'suites.json').write_text(json.dumps({DEFAULT_SUITE: malformed}), encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'unique task'):
                    suite_tasks(DEFAULT_SUITE, root)
            (directory / 'suites.json').write_text(json.dumps({DEFAULT_SUITE: correct}), encoding='utf-8')
            registry.pop(correct[-1])
            (directory / 'tasks.json').write_text(json.dumps(registry), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'absent from the registry'):
                suite_tasks(DEFAULT_SUITE, root)

    def test_legacy_recipe_is_unchanged_and_independent(self):
        original = template()
        selected = resolve_suite(original, root=ROOT)
        self.assertEqual(selected, original)
        selected['tasks'].append('gam_lvis')
        self.assertNotEqual(selected['tasks'], original['tasks'])
        renamed = resolve_suite(original, root=ROOT, run_id='explicit_smoke')
        self.assertEqual(renamed, {**original, 'run_id': 'explicit_smoke'})

    def test_evaluator_plan_accepts_yaml_suite_and_preserves_full_scope(self):
        config = template()
        config.pop('tasks')
        config.update(suite=DEFAULT_SUITE, limit=None, run_id='suite_plan_only')
        prepared = evaluation_plan(config, root=ROOT)
        self.assertEqual(prepared['suite'], DEFAULT_SUITE)
        self.assertEqual(prepared['evaluation_scope'], 'full')
        self.assertEqual(prepared['tasks'], suite_tasks(DEFAULT_SUITE, ROOT))
        self.assertNotIn('GAM_EVAL_LIMIT', prepared['env'])
        environment = evaluation_environment(prepared, root=ROOT, inherited={'GAM_EVAL_LIMIT': '8'})
        self.assertNotIn('GAM_EVAL_LIMIT', environment)
        self.assertEqual(prepared['mode'], config['mode'])
        self.assertEqual(prepared['effective_runtime']['decoder'], config.get('decoder'))
        self.assertNotIn('tasks', config)


class LauncherIntegrationTests(unittest.TestCase):
    def command(self, *arguments):
        args = cli.parser_for(ROOT).parse_args(['eval', *arguments, '--dry-run'])
        return cli.plan(args, ROOT)['command']

    def test_launcher_forwards_explicit_suite_and_run_id(self):
        command = self.command('--suite', DEFAULT_SUITE, '--run-id', 'comparison_02')
        self.assertEqual(command[-4:], ['--suite', DEFAULT_SUITE, '--run-id', 'comparison_02'])
        self.assertIn('configs/eval/' + DEFAULT_TEMPLATE, command)
        self.assertIn('.venv-eval', command[0])

    def test_launcher_preserves_existing_and_explicit_template_routes(self):
        self.assertEqual(self.command()[-1], 'configs/eval/' + DEFAULT_TEMPLATE)
        command = self.command('--config', 'configs/eval/vlm.yaml', '--suite', DEFAULT_SUITE)
        self.assertIn('configs/eval/vlm.yaml', command)
        if IS_GROUNDANYTHING:
            command = self.command('--decoder', 'speculative', '--suite', DEFAULT_SUITE)
            self.assertIn('configs/eval/dlm_speculative.yaml', command)

    def test_actual_evaluator_dry_run_uses_full_scope_without_contacting_service(self):
        result = subprocess.run([sys.executable, '-X', 'utf8', str(ROOT / 'scripts/evaluate.py'),
                                 'configs/eval/' + DEFAULT_TEMPLATE, '--suite', DEFAULT_SUITE,
                                 '--run-id', 'suite_cli_dry_run', '--dry-run'],
                                cwd=ROOT, capture_output=True, text=True, encoding='utf-8', timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        prepared = json.loads(result.stdout)
        self.assertEqual(prepared['suite'], DEFAULT_SUITE)
        self.assertEqual(len(prepared['tasks']), SUITE_COUNTS[DEFAULT_SUITE])
        self.assertEqual(prepared['evaluation_scope'], 'full')
        self.assertEqual(Path(prepared['output']).name, 'suite_cli_dry_run')
        self.assertNotIn('GAM_EVAL_LIMIT', prepared['env'])


if __name__ == '__main__':
    unittest.main()

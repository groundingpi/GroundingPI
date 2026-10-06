"""Resolve the published benchmark suites without changing inference settings."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import uuid

ROOT = Path(__file__).resolve().parents[1]
SUITE_COUNTS = {'groundanything30': 30, 'groundingpi34': 34}


def suite_tasks(name, root=ROOT):
    """Return a validated, ordered task list from the release suite inventory."""
    if not isinstance(name, str) or name not in SUITE_COUNTS:
        raise ValueError('unknown evaluation suite; choose ' + ', '.join(SUITE_COUNTS))
    root = Path(root)
    suites = json.loads((root / 'configs/eval/suites.json').read_text(encoding='utf-8'))
    tasks = suites.get(name)
    if (not isinstance(tasks, list)
            or any(not isinstance(task, str) for task in tasks)
            or len(tasks) != SUITE_COUNTS[name] or len(set(tasks)) != len(tasks)):
        raise ValueError(f'{name} must contain exactly {SUITE_COUNTS[name]} unique task names')
    registry = json.loads((root / 'configs/eval/tasks.json').read_text(encoding='utf-8'))
    unknown = sorted(set(tasks) - set(registry))
    if unknown:
        raise ValueError(f'{name} contains tasks absent from the registry: {unknown}')
    return list(tasks)


def resolve_suite(config, root=ROOT, suite=None, run_id=None):
    """Return an independent effective recipe.

    A CLI suite explicitly replaces a template's task selection and sample limit,
    and generates a fresh run ID unless one is supplied. A YAML suite declares a
    full run: conflicting tasks or a finite sample limit are rejected. Legacy
    task-only recipes retain their behavior. Model, service, mode, decoder and
    output-token settings are never changed by suite selection.
    """
    if not isinstance(config, dict):
        raise ValueError('evaluation configuration must be a mapping')
    result = deepcopy(config)
    selected = suite if suite is not None else result.get('suite')
    if selected is None and 'suite' in result:
        raise ValueError('suite must name a published evaluation suite')
    if selected is not None:
        tasks = suite_tasks(selected, root)
        if suite is None:
            if result.get('limit') is not None:
                raise ValueError('a named suite is a full evaluation; set limit: null or use a task-only smoke recipe')
            if 'tasks' in result and result['tasks'] != tasks:
                raise ValueError('tasks conflict with the named suite; omit tasks or use its exact ordered task list')
        result.update(suite=selected, tasks=tasks, limit=None)
        if suite is not None and run_id is None:
            stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
            result['run_id'] = f'{selected}_{stamp}_{uuid.uuid4().hex[:8]}'
    if run_id is not None:
        if not isinstance(run_id, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', run_id):
            raise ValueError('run_id must be a safe unique directory name')
        result['run_id'] = run_id
    return result

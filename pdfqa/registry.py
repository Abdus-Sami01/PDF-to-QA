"""Extension points: custom synthesis tasks, export formats, and input adapters.

Registration is additive — built-ins are untouched and keep working whether or not anything is
registered. A plugin is an ordinary Python module that imports this and decorates a function; point
`plugins` in the config at it and the pipeline picks it up.
"""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from typing import Callable

TASKS: dict[str, Callable] = {}
FORMATS: dict[str, Callable] = {}
ADAPTERS: dict[str, Callable] = {}
_LOADED: set[str] = set()


def register_task(name: str) -> Callable:
    """Signature: fn(runtime, chunks, kg, config) -> list[QARecord]. Runs after the built-in tasks."""

    def wrap(fn: Callable) -> Callable:
        TASKS[name] = fn
        return fn

    return wrap


def register_format(name: str) -> Callable:
    """Signature: fn(records) -> list[dict]. Rows are written to <outdir>/<name>.jsonl."""

    def wrap(fn: Callable) -> Callable:
        FORMATS[name] = fn
        return fn

    return wrap


def register_adapter(*suffixes: str) -> Callable:
    """Signature: fn(path) -> DocumentTree. Suffixes include the dot."""

    def wrap(fn: Callable) -> Callable:
        for suffix in suffixes:
            ADAPTERS[suffix.lower()] = fn
        return fn

    return wrap


def load_plugins(specs: list[str]) -> list[str]:
    """Accepts importable module names or paths to .py files; importing is what registers things."""
    loaded = []
    for spec in specs:
        if spec in _LOADED:
            continue
        path = Path(spec)
        if path.suffix == ".py" and path.exists():
            module_spec = importlib.util.spec_from_file_location(path.stem, path)
            if module_spec is None or module_spec.loader is None:
                raise ImportError(f"cannot load plugin from {spec}")
            module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)
        else:
            importlib.import_module(spec)
        _LOADED.add(spec)
        loaded.append(spec)
    return loaded


def clear() -> None:
    TASKS.clear()
    FORMATS.clear()
    ADAPTERS.clear()
    _LOADED.clear()


def summary() -> dict:
    return {"tasks": sorted(TASKS), "formats": sorted(FORMATS), "adapters": sorted(ADAPTERS), "plugins": sorted(_LOADED)}

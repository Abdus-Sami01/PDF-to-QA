"""Content-addressed stage cache so parser or prompt edits only invalidate what they touch."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

SCHEMA = 2


def fingerprint(*parts: Any) -> str:
    h = hashlib.blake2b(digest_size=16)
    for p in parts:
        if isinstance(p, (dict, list, tuple)):
            p = json.dumps(p, sort_keys=True, default=str)
        h.update(str(p).encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()


def file_fingerprint(path: str | Path) -> str:
    p = Path(path)
    h = hashlib.blake2b(digest_size=16)
    with p.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class Store:
    """One directory per stage; each entry is a JSON blob keyed by its input fingerprint."""

    def __init__(self, root: str | Path = ".pdfqa-cache", enabled: bool = True):
        self.root = Path(root)
        self.enabled = enabled
        self.hits = 0
        self.misses = 0
        if enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    def path(self, stage: str, key: str) -> Path:
        return self.root / stage / f"{key}.json"

    def get(self, stage: str, key: str):
        if not self.enabled:
            return None
        p = self.path(stage, key)
        payload = None
        if p.exists():
            try:
                payload = json.loads(p.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                payload = None
        if payload is None or payload.get("schema") != SCHEMA:
            self.misses += 1
            return None
        self.hits += 1
        return payload["value"]

    def put(self, stage: str, key: str, value) -> None:
        if not self.enabled:
            return
        p = self.path(stage, key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps({"schema": SCHEMA, "stage": stage, "key": key, "at": time.time(), "value": value}, ensure_ascii=False, default=str), encoding="utf-8")
        tmp.replace(p)

    def memoize(self, stage: str, key: str, fn: Callable[[], Any]):
        cached = self.get(stage, key)
        if cached is not None:
            return cached
        value = fn()
        self.put(stage, key, value)
        return value

    def invalidate(self, stage: str | None = None) -> int:
        target = self.root / stage if stage else self.root
        if not target.exists():
            return 0
        n = 0
        for p in target.rglob("*.json"):
            p.unlink()
            n += 1
        return n

    def stats(self) -> dict:
        stages = {}
        if self.root.exists():
            for d in sorted(self.root.iterdir()):
                if d.is_dir():
                    stages[d.name] = len(list(d.glob("*.json")))
        return {"root": str(self.root), "enabled": self.enabled, "hits": self.hits, "misses": self.misses, "entries": stages}

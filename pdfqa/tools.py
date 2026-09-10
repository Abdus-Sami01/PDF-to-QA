"""Executable tools behind ReAct traces: sandboxed Python, SQL over extracted tables, source lookup.

The same executors both replay a generated trace during verification and can drive a live agent,
so a trace that passes verification is one whose observations were actually produced, not asserted.
"""

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass

NUMBER = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?")
"""Thousands separators included, because a paper writes 1,000 and an answer writes 1000.

Matching bare digit runs split "1,000" into 1 and 000, which broke grounding in both directions: a
correct answer saying 1000 was rejected as unsupported, and a hallucinated 1,500 was accepted
against a source saying 1,000 because both reduced to fragments the other happened to contain.
"""


def numbers_in(text: str) -> list[float]:
    """Every number in the text, thousands separators removed before conversion."""
    return [float(m.replace(",", "")) for m in NUMBER.findall(text)]


IDENT = re.compile(r"[^a-z0-9_]+")
SELECT_ONLY = re.compile(r"^\s*(select|with)\b", re.I)
SQL_FORBIDDEN = re.compile(r"\b(attach|pragma|insert|update|delete|drop|alter|create|vacuum|load_extension)\b", re.I)


@dataclass
class ToolResult:
    ok: bool
    output: str
    tool: str = ""

    def __str__(self) -> str:
        return self.output


def column_name(raw: str, index: int, taken: set[str]) -> str:
    name = IDENT.sub("_", raw.strip().lower()).strip("_")
    if not name or name[0].isdigit():
        name = f"c{index}"
    base, n = name, 2
    while name in taken:
        name, n = f"{base}_{n}", n + 1
    taken.add(name)
    return name


def _cast(value: str):
    text = value.strip()
    if not text:
        return None
    cleaned = text.replace(",", "").rstrip("%")
    try:
        return int(cleaned)
    except ValueError:
        pass
    try:
        return float(cleaned)
    except ValueError:
        return text


def table_name(caption: str, index: int, taken: set[str]) -> str:
    """`t` is always the first table; captions also get a readable alias like table_1."""
    if index == 0:
        taken.add("t")
        return "t"
    m = re.match(r"\s*(table|figure)\s*(\d{1,3}[a-z]?)", caption.strip(), re.I)
    if m:
        name = f"{m.group(1).lower()}_{m.group(2).lower()}"
        if name not in taken:
            taken.add(name)
            return name
    name = f"t{index + 1}"
    taken.add(name)
    return name


def load_tables(grids: list[list[list[str]]], captions: list[str] | None = None) -> tuple[sqlite3.Connection, dict[str, list[str]]]:
    """Header row becomes columns; returns the connection plus the schema actually created."""
    conn = sqlite3.connect(":memory:")
    captions = captions or []
    schema: dict[str, list[str]] = {}
    taken_names: set[str] = set()
    for i, grid in enumerate(grids):
        if len(grid) < 2:
            continue
        name = table_name(captions[i] if i < len(captions) else "", len(schema), taken_names)
        taken: set[str] = set()
        cols = [column_name(c, j, taken) for j, c in enumerate(grid[0])]
        conn.execute(f"CREATE TABLE {name} ({', '.join(cols)})")
        placeholders = ",".join("?" * len(cols))
        for row in grid[1:]:
            padded = (list(row) + [""] * len(cols))[: len(cols)]
            conn.execute(f"INSERT INTO {name} VALUES ({placeholders})", [_cast(c) for c in padded])
        schema[name] = cols
    conn.commit()
    return conn, schema


def describe_schema(grids: list[list[list[str]]], captions: list[str] | None = None) -> str:
    """The model cannot write valid SQL without knowing the tables; this goes into the prompt."""
    if not grids:
        return "(no tables in scope)"
    conn, schema = load_tables(grids, captions)
    conn.close()
    return "\n".join(f"{name}({', '.join(cols)})" for name, cols in schema.items()) or "(no tables in scope)"


def run_sql(query: str, grids: list[list[list[str]]], limit: int = 50, captions: list[str] | None = None,
            timeout: float = 10.0) -> ToolResult:
    if SQL_FORBIDDEN.search(query) or not SELECT_ONLY.match(query):
        return ToolResult(False, "only read-only SELECT queries are allowed", "sql")
    if not grids:
        return ToolResult(False, "no table available in this context", "sql")
    try:
        conn, _ = load_tables(grids, captions)
    except sqlite3.Error as exc:
        return ToolResult(False, f"table load failed: {exc}", "sql")
    # `WITH RECURSIVE forever(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM forever)` is a legal SELECT
    # that never returns. Without this the worker thread replaying the trace hangs for good, and
    # enough of them stall the whole run.
    deadline = time.monotonic() + timeout
    conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10000)
    try:
        rows = conn.execute(query).fetchmany(limit)
    except sqlite3.Error as exc:
        expired = time.monotonic() > deadline
        return ToolResult(False, f"query exceeded {timeout:g}s and was cancelled" if expired else f"sql error: {exc}", "sql")
    finally:
        conn.set_progress_handler(None, 0)
        conn.close()
    if not rows:
        return ToolResult(True, "(no rows)", "sql")
    return ToolResult(True, "\n".join(" | ".join("" if c is None else str(c) for c in row) for row in rows), "sql")


def run_lookup(term: str, context: str, window: int = 320) -> ToolResult:
    needle = term.strip().lower()
    if not needle:
        return ToolResult(False, "empty lookup term", "lookup")
    hay = context.lower()
    i = hay.find(needle)
    if i == -1:
        words = [w for w in re.findall(r"[a-z0-9]+", needle) if len(w) > 2]
        i = min((hay.find(w) for w in words if hay.find(w) != -1), default=-1)
    if i == -1:
        return ToolResult(True, "(not found in source)", "lookup")
    start = max(0, i - window // 2)
    return ToolResult(True, context[start : start + window].strip(), "lookup")


def execute(
    action: str,
    action_input: str,
    context: str = "",
    grids: list[list[list[str]]] | None = None,
    timeout: float = 10.0,
    captions: list[str] | None = None,
) -> ToolResult:
    from .verify import run_python

    action = (action or "").strip().lower()
    if action == "python":
        ok, out = run_python(action_input, timeout)
        return ToolResult(ok, out, "python")
    if action == "sql":
        return run_sql(action_input, grids or [], captions=captions, timeout=timeout)
    if action == "lookup":
        return run_lookup(action_input, context)
    return ToolResult(False, f"unknown tool: {action}", action)


def observations_agree(claimed: str, actual: str, tolerance: float = 0.01) -> bool:
    """Numbers must match within tolerance; otherwise fall back to containment either way."""
    claimed, actual = (claimed or "").strip(), (actual or "").strip()
    if not claimed:
        return False
    c_nums = numbers_in(claimed)
    a_nums = numbers_in(actual)
    if c_nums and a_nums:
        return all(any(abs(c - a) <= max(tolerance, abs(a) * tolerance) for a in a_nums) for c in c_nums)
    low_c, low_a = claimed.lower(), actual.lower()
    return low_c in low_a or low_a in low_c

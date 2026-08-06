"""Pipeline configuration. Load from YAML or JSON; every stage can be toggled independently."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path

PARSER_VERSION = "6"
PROMPT_VERSION = "6"


@dataclass
class ChunkConfig:
    max_tokens: int = 900
    min_tokens: int = 60
    overlap_headings: bool = True
    dedup_threshold: float = 0.85


@dataclass
class GraphConfig:
    enabled: bool = True
    llm_extract: bool = True
    max_chunks: int | None = None


@dataclass
class SynthConfig:
    qa_per_chunk: int = 3
    figure_qa_per_doc: int = 4
    multihop_pairs: int = 8
    multihop_per_pair: int = 1
    cross_doc_pairs: int = 6
    multiturn_per_doc: int = 4
    multiturn_length: int = 6
    react_per_doc: int = 3
    repair_traces: bool = True
    persona_ratio: float = 0.3
    evol_ratio: float = 0.25
    dpo_ratio: float = 0.4
    personas: list[str] = field(default_factory=lambda: ["domain_expert", "non_technical_stakeholder", "student", "skeptical_reviewer", "practitioner"])
    styles: list[str] = field(default_factory=lambda: ["prose", "bullets", "json", "step_by_step"])
    temperature: float = 0.8


@dataclass
class VerifyConfig:
    enabled: bool = True
    model_gates: bool = True
    symbolic: bool = True
    z3: bool = True
    allow_exec: bool = True
    consistency_samples: int = 0
    min_quality: float = 0.55


@dataclass
class SelectConfig:
    lexical_threshold: float = 0.8
    against: list[str] = field(default_factory=list)
    semantic_threshold: float = 0.92
    dpp: bool = True
    budget_tokens: int | None = None
    target_count: int | None = None
    balance: bool = True
    mix: dict[str, float] = field(default_factory=lambda: {"simple": 0.25, "intermediate": 0.45, "complex": 0.30})


@dataclass
class RuntimeConfig:
    generate: dict = field(default_factory=lambda: {"backend": "echo"})
    verify: dict = field(default_factory=lambda: {"backend": "echo"})
    embed: dict = field(default_factory=dict)
    vision: dict = field(default_factory=dict)
    workers: int = 4
    retries: int = 2


@dataclass
class Config:
    inputs: list[str] = field(default_factory=list)
    outdir: str = "out"
    formats: list[str] = field(default_factory=lambda: ["chatml", "sharegpt", "dpo", "raw"])
    cache_dir: str = ".pdfqa-cache"
    cache: bool = True
    assets_dir: str | None = "assets"
    corpus: bool = True
    split: list[float] = field(default_factory=list)
    figure_dpi: int = 144
    seed: int = 7
    backend: str = "auto"
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    synth: SynthConfig = field(default_factory=SynthConfig)
    verify: VerifyConfig = field(default_factory=VerifyConfig)
    select: SelectConfig = field(default_factory=SelectConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def runtime_specs(self) -> dict[str, dict]:
        specs = {"generate": dict(self.runtime.generate), "verify": dict(self.runtime.verify or self.runtime.generate)}
        if self.runtime.embed:
            specs["embed"] = dict(self.runtime.embed)
        if self.runtime.vision:
            specs["vision"] = dict(self.runtime.vision)
        return specs

    def as_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def load(path: str | Path) -> "Config":
        p = Path(path)
        text = p.read_text(encoding="utf-8")
        if p.suffix.lower() in (".yaml", ".yml"):
            import yaml  # type: ignore

            data = yaml.safe_load(text) or {}
        else:
            data = json.loads(text)
        return Config.from_dict(data)

    @staticmethod
    def from_dict(data: dict) -> "Config":
        return _build(Config, data)


def _build(cls, data: dict):
    kwargs = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        if is_dataclass(f.type) and isinstance(value, dict):
            kwargs[f.name] = _build(f.type, value)
        elif isinstance(value, dict) and f.name in _NESTED:
            kwargs[f.name] = _build(_NESTED[f.name], value)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)


_NESTED = {
    "chunk": ChunkConfig,
    "graph": GraphConfig,
    "synth": SynthConfig,
    "verify": VerifyConfig,
    "select": SelectConfig,
    "runtime": RuntimeConfig,
}

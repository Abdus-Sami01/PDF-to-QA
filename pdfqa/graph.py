"""Document-level knowledge graph over chunks, with cross-chunk path search.

Nodes are entities (methods, datasets, metrics, quantities, claims); edges come from
LLM extraction, co-occurrence within a chunk, and the AST reference graph.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field

from .llm import Runtime, parse_json
from .prompts import KG_EXTRACT
from .records import Chunk

ENTITY_TYPES = ("method", "dataset", "metric", "quantity", "claim", "entity", "hyperparameter", "term")
GENERIC_TYPES = ("entity", "term")

QUANTITY_RE = re.compile(
    r"(?<![\w.])(-?\d+(?:\.\d+)?(?:[eE]-?\d+)?)\s*(%|percent|ms|s\b|GB|MB|K\b|M\b|B\b|x\b|×|FLOPs|tokens?|epochs?|layers?|steps?)?",
)


@dataclass
class Entity:
    name: str
    type: str = "entity"
    chunk_ids: set[str] = field(default_factory=set)
    node_ids: set[str] = field(default_factory=set)
    mentions: list[str] = field(default_factory=list)
    aliases: set[str] = field(default_factory=set)
    value: str = ""

    @property
    def key(self) -> str:
        return normalize(self.name)


@dataclass
class Edge:
    src: str
    dst: str
    relation: str
    chunk_id: str = ""
    weight: float = 1.0


def normalize(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower()).strip(" .,:;()[]")


class KnowledgeGraph:
    def __init__(self):
        self.entities: dict[str, Entity] = {}
        self.edges: list[Edge] = []
        self.adj: dict[str, list[Edge]] = defaultdict(list)
        self.chunk_index: dict[str, Chunk] = {}
        self.meta: dict = {}

    def add_entity(self, name: str, etype: str = "entity", chunk: Chunk | None = None, value: str = "") -> Entity | None:
        key = normalize(name)
        if len(key) < 2 or len(key) > 120:
            return None
        ent = self.entities.get(key)
        if ent is None:
            ent = Entity(name=name.strip(), type=etype if etype in ENTITY_TYPES else "entity", value=value)
            self.entities[key] = ent
        if value and not ent.value:
            ent.value = value
        if ent.type in GENERIC_TYPES and etype in ENTITY_TYPES and etype not in GENERIC_TYPES:
            ent.type = etype
        if chunk is not None:
            ent.chunk_ids.add(chunk.id)
            ent.node_ids.update(chunk.prov.node_ids)
        return ent

    def add_edge(self, src: str, dst: str, relation: str, chunk_id: str = "", weight: float = 1.0) -> None:
        s, d = normalize(src), normalize(dst)
        if s == d or s not in self.entities or d not in self.entities:
            return
        edge = Edge(s, d, relation, chunk_id, weight)
        self.edges.append(edge)
        self.adj[s].append(edge)
        self.adj[d].append(Edge(d, s, f"inv:{relation}", chunk_id, weight))

    def neighbors(self, key: str) -> list[Edge]:
        return self.adj.get(normalize(key), [])

    def paths(self, src: str, dst: str, max_hops: int = 3) -> list[list[Edge]]:
        src, dst = normalize(src), normalize(dst)
        results: list[list[Edge]] = []
        stack = [(src, [], {src})]
        while stack:
            node, path, seen = stack.pop()
            if len(path) >= max_hops:
                continue
            for edge in self.adj.get(node, []):
                if edge.dst in seen:
                    continue
                new_path = path + [edge]
                if edge.dst == dst:
                    results.append(new_path)
                    continue
                stack.append((edge.dst, new_path, seen | {edge.dst}))
        results.sort(key=len)
        return results[:16]

    def bridging_pairs(self, min_gap: int = 1, limit: int = 64) -> list[tuple[Entity, Entity, list[Edge]]]:
        """Entity pairs that are graph-connected but live in different chunks — multi-hop seeds."""
        out = []
        keys = list(self.entities)
        for i, a in enumerate(keys):
            ea = self.entities[a]
            for b in keys[i + 1 :]:
                eb = self.entities[b]
                if ea.chunk_ids & eb.chunk_ids:
                    continue
                if not ea.chunk_ids or not eb.chunk_ids:
                    continue
                path = self.paths(a, b, max_hops=3)
                if path and len(path[0]) > min_gap:
                    out.append((ea, eb, path[0]))
        out.sort(key=lambda t: (len(t[2]), -len(t[0].chunk_ids | t[1].chunk_ids)))
        return out[:limit]

    def chunks_for(self, *entity_keys: str) -> list[Chunk]:
        ids: set[str] = set()
        for k in entity_keys:
            ent = self.entities.get(normalize(k))
            if ent:
                ids |= ent.chunk_ids
        return [self.chunk_index[i] for i in ids if i in self.chunk_index]

    def merge_entities(self, keep: str, drop: str) -> None:
        if keep == drop or keep not in self.entities or drop not in self.entities:
            return
        target, other = self.entities[keep], self.entities.pop(drop)
        target.chunk_ids |= other.chunk_ids
        target.node_ids |= other.node_ids
        target.mentions.extend(other.mentions)
        target.aliases.add(other.name)
        target.aliases |= other.aliases
        if not target.value:
            target.value = other.value
        if target.type in GENERIC_TYPES and other.type not in GENERIC_TYPES:
            target.type = other.type
        self._reindex(drop, keep)

    def _reindex(self, old: str, new: str) -> None:
        self.adj.clear()
        edges, seen = [], set()
        for e in self.edges:
            src = new if e.src == old else e.src
            dst = new if e.dst == old else e.dst
            if src == dst or src not in self.entities or dst not in self.entities:
                continue
            sig = (src, dst, e.relation)
            if sig in seen:
                continue
            seen.add(sig)
            edges.append(Edge(src, dst, e.relation, e.chunk_id, e.weight))
        self.edges = []
        for e in edges:
            self.add_edge(e.src, e.dst, e.relation, e.chunk_id, e.weight)

    def canonicalise(self) -> int:
        """Fold surface variants together: punctuation and spacing, plurals, and acronyms."""
        merges = 0
        for group in _variant_groups(list(self.entities)).values():
            if len(group) < 2:
                continue
            keep = max(group, key=lambda k: (len(self.entities[k].chunk_ids), -len(k)))
            for other in group:
                if other != keep and other in self.entities:
                    self.merge_entities(keep, other)
                    merges += 1
        for short, long in _acronym_pairs(self.entities).items():
            if short in self.entities and long in self.entities:
                self.merge_entities(long, short)
                merges += 1
        return merges

    def stats(self) -> dict:
        by_type: dict[str, int] = defaultdict(int)
        for e in self.entities.values():
            by_type[e.type] += 1
        return {"entities": len(self.entities), "edges": len(self.edges), "by_type": dict(by_type)}

    def as_dict(self) -> dict:
        return {
            "entities": [
                {
                    "name": e.name,
                    "type": e.type,
                    "value": e.value,
                    "aliases": sorted(e.aliases),
                    "chunk_ids": sorted(e.chunk_ids),
                    "node_ids": sorted(e.node_ids),
                }
                for e in self.entities.values()
            ],
            "edges": [e.__dict__ for e in self.edges],
        }

    @staticmethod
    def from_dict(data: dict, chunks: list[Chunk] | None = None) -> "KnowledgeGraph":
        kg = KnowledgeGraph()
        for e in data.get("entities", []):
            ent = Entity(name=e["name"], type=e.get("type", "entity"), value=e.get("value", ""))
            ent.chunk_ids = set(e.get("chunk_ids", []))
            ent.node_ids = set(e.get("node_ids", []))
            ent.aliases = set(e.get("aliases", []))
            kg.entities[ent.key] = ent
        for e in data.get("edges", []):
            kg.add_edge(e["src"], e["dst"], e["relation"], e.get("chunk_id", ""), e.get("weight", 1.0))
        for c in chunks or []:
            kg.chunk_index[c.id] = c
        return kg


def _collapse(key: str) -> str:
    base = re.sub(r"[^a-z0-9]", "", key)
    return base[:-1] if len(base) > 4 and base.endswith("s") and not base.endswith("ss") else base


def _variant_groups(keys: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for k in keys:
        collapsed = _collapse(k)
        if len(collapsed) >= 4:
            groups[collapsed].append(k)
    return groups


def _acronym_pairs(entities: dict[str, Entity]) -> dict[str, str]:
    """Map an all-caps key like `srn` onto `sparse routing network` when the initials line up."""
    initials: dict[str, str] = {}
    for key, ent in entities.items():
        words = [w for w in re.split(r"[\s\-]+", key) if w]
        if len(words) < 2:
            continue
        acronym = "".join(w[0] for w in words)
        if len(acronym) >= 2:
            initials.setdefault(acronym, key)
    out: dict[str, str] = {}
    for key, ent in entities.items():
        if " " in key or len(key) < 2 or len(key) > 6:
            continue
        if ent.name.isupper() or key in initials:
            long = initials.get(key)
            if long and long != key:
                out[key] = long
    return out


def build_graph(chunks: list[Chunk], runtime: Runtime | None = None, llm_extract: bool = True, max_chunks: int | None = None) -> KnowledgeGraph:
    kg = KnowledgeGraph()
    for c in chunks:
        kg.chunk_index[c.id] = c

    target = chunks if max_chunks is None else chunks[:max_chunks]
    for chunk in target:
        symbolic_pass(kg, chunk)
        if llm_extract and runtime is not None:
            llm_pass(kg, chunk, runtime)
    merged = kg.canonicalise()
    link_shared_sections(kg, chunks)
    if merged:
        kg.meta["canonical_merges"] = merged
    return kg


def symbolic_pass(kg: KnowledgeGraph, chunk: Chunk) -> None:
    """Cheap, deterministic seeds: quantities, capitalised terms, and breadcrumb anchors."""
    for m in QUANTITY_RE.finditer(chunk.text):
        raw = m.group(0).strip()
        if not raw or len(raw) < 2 or raw.isalpha():
            continue
        context = chunk.text[max(0, m.start() - 60) : m.start()].strip().split("\n")[-1]
        label = f"{context[-40:]} {raw}".strip() if context else raw
        ent = kg.add_entity(label, "quantity", chunk, value=raw)
        if ent:
            ent.mentions.append(raw)
    for m in re.finditer(r"\b([A-Z][A-Za-z0-9]*(?:[- ][A-Z][A-Za-z0-9]*){0,3})\b", chunk.text):
        term = m.group(1)
        if len(term) < 3 or term.lower() in STOP_TERMS:
            continue
        kg.add_entity(term, "term", chunk)
    for part in chunk.prov.section_path:
        kg.add_entity(part, "term", chunk)


STOP_TERMS = {"the", "this", "that", "these", "those", "we", "our", "in", "as", "for", "table", "figure", "section", "however", "moreover", "thus"}


def llm_pass(kg: KnowledgeGraph, chunk: Chunk, runtime: Runtime) -> None:
    prompt = KG_EXTRACT.format(breadcrumb=chunk.prov.breadcrumb, context=chunk.context()[:6000])
    try:
        raw = runtime.complete("generate", prompt, temperature=0.0, max_tokens=1400)
    except Exception:
        return
    data = parse_json(raw, default={}) or {}
    if isinstance(data, list):
        data = {"entities": data, "relations": []}
    for e in data.get("entities", []) or []:
        if isinstance(e, str):
            kg.add_entity(e, "entity", chunk)
        elif isinstance(e, dict) and e.get("name"):
            kg.add_entity(e["name"], str(e.get("type", "entity")).lower(), chunk, value=str(e.get("value", "")))
    for r in data.get("relations", []) or []:
        if not isinstance(r, dict):
            continue
        src, dst = r.get("source") or r.get("src"), r.get("target") or r.get("dst")
        if not src or not dst:
            continue
        kg.add_entity(src, "entity", chunk)
        kg.add_entity(dst, "entity", chunk)
        kg.add_edge(src, dst, str(r.get("relation", "related_to")), chunk.id)


def link_shared_sections(kg: KnowledgeGraph, chunks: list[Chunk]) -> None:
    """Co-occurrence edges within a chunk, and section-level bridges across chunks."""
    per_chunk: dict[str, list[str]] = defaultdict(list)
    for key, ent in kg.entities.items():
        for cid in ent.chunk_ids:
            per_chunk[cid].append(key)
    for cid, keys in per_chunk.items():
        salient = sorted(keys, key=lambda k: (kg.entities[k].type == "quantity", len(kg.entities[k].chunk_ids)), reverse=True)[:12]
        for i, a in enumerate(salient):
            for b in salient[i + 1 :]:
                kg.add_edge(a, b, "co_occurs", cid, 0.4)
    by_section: dict[str, list[str]] = defaultdict(list)
    for chunk in chunks:
        root = chunk.prov.section_path[0] if chunk.prov.section_path else ""
        if root:
            by_section[root].append(chunk.id)
    for section, cids in by_section.items():
        anchors = [k for k, e in kg.entities.items() if len(e.chunk_ids & set(cids)) > 1]
        for i, a in enumerate(anchors[:10]):
            for b in anchors[i + 1 : 10]:
                kg.add_edge(a, b, "same_section", "", 0.2)


def merge_graphs(graphs: list[KnowledgeGraph], chunks: list[Chunk] | None = None) -> KnowledgeGraph:
    """One graph over the whole corpus, so an entity discussed in two papers becomes a single node."""
    corpus = KnowledgeGraph()
    for c in chunks or []:
        corpus.chunk_index[c.id] = c
    for kg in graphs:
        corpus.chunk_index.update(kg.chunk_index)
        for key, ent in kg.entities.items():
            merged = corpus.entities.get(key)
            if merged is None:
                merged = Entity(name=ent.name, type=ent.type, value=ent.value)
                corpus.entities[key] = merged
            merged.chunk_ids |= ent.chunk_ids
            merged.node_ids |= ent.node_ids
            merged.aliases |= ent.aliases
            if merged.type in GENERIC_TYPES and ent.type not in GENERIC_TYPES:
                merged.type = ent.type
        for e in kg.edges:
            corpus.add_edge(e.src, e.dst, e.relation, e.chunk_id, e.weight)
    corpus.meta["canonical_merges"] = corpus.canonicalise()
    corpus.meta["documents"] = len(graphs)
    return corpus


def source_of(kg: KnowledgeGraph, ent: Entity) -> set[str]:
    return {kg.chunk_index[c].prov.source for c in ent.chunk_ids if c in kg.chunk_index}


def cross_document_pairs(kg: KnowledgeGraph, limit: int = 32) -> list[tuple[Entity, Entity, list[Edge]]]:
    """Seeds where the two supporting chunks can come from different documents.

    Two shapes qualify: one concept discussed in several documents (the strongest signal, and the
    one an entity-merged corpus graph exists to find), and an edge whose endpoints never co-occur
    in the same source.
    """
    out: list[tuple[Entity, Entity, list[Edge]]] = []
    seen: set[tuple[str, str]] = set()

    for ent in kg.entities.values():
        if len(source_of(kg, ent)) > 1:
            out.append((ent, ent, []))
            seen.add((ent.key, ent.key))

    for edge in kg.edges:
        a, b = kg.entities.get(edge.src), kg.entities.get(edge.dst)
        if a is None or b is None:
            continue
        sig = tuple(sorted((a.key, b.key)))
        if sig in seen:
            continue
        sa, sb = source_of(kg, a), source_of(kg, b)
        if sa and sb and not (sa & sb):
            seen.add(sig)
            out.append((a, b, [edge]))

    out.sort(key=lambda t: (-len(source_of(kg, t[0]) | source_of(kg, t[1])), -len(t[0].chunk_ids)))
    return out[:limit]


def serialize_path(kg: KnowledgeGraph, path: list[Edge]) -> str:
    if not path:
        return ""
    parts = [kg.entities[path[0].src].name]
    for edge in path:
        parts.append(f"-[{edge.relation}]-> {kg.entities[edge.dst].name}")
    return " ".join(parts)


def dump(kg: KnowledgeGraph) -> str:
    return json.dumps(kg.as_dict(), ensure_ascii=False, indent=2)

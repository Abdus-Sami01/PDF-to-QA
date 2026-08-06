"""Model runtime: local (Ollama / vLLM / any OpenAI-compatible server) and cloud providers.

Roles let one pipeline route cheap bulk generation locally and route verification to a
stronger cloud model without any call site knowing which backend it landed on.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable

ROLES = ("generate", "verify", "embed")


class LLMError(RuntimeError):
    pass


@dataclass
class Completion:
    text: str
    model: str
    latency: float = 0.0
    usage: dict = field(default_factory=dict)


class Backend:
    name = "base"

    def complete(self, prompt: str, system: str = "", temperature: float = 0.7, max_tokens: int = 1024) -> Completion:
        raise NotImplementedError

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError(f"{self.name} has no embedding endpoint")


def _post(url: str, payload: dict, headers: dict, timeout: float) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise LLMError(f"{url} -> {exc.code}: {exc.read()[:500].decode('utf-8', 'replace')}") from exc
    except urllib.error.URLError as exc:
        raise LLMError(f"{url} unreachable: {exc.reason}") from exc


class OpenAICompatible(Backend):
    """Covers OpenAI, vLLM, llama.cpp server, LM Studio, and anything else speaking /v1/chat/completions."""

    name = "openai"

    def __init__(self, model: str, base_url: str = "https://api.openai.com/v1", api_key: str = "", timeout: float = 180.0):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.timeout = timeout

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def complete(self, prompt, system="", temperature=0.7, max_tokens=1024):
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        t0 = time.time()
        data = _post(
            f"{self.base_url}/chat/completions",
            {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens},
            self._headers(),
            self.timeout,
        )
        return Completion(data["choices"][0]["message"]["content"], self.model, time.time() - t0, data.get("usage", {}))

    def embed(self, texts):
        data = _post(f"{self.base_url}/embeddings", {"model": self.model, "input": texts}, self._headers(), self.timeout)
        return [d["embedding"] for d in data["data"]]


class Anthropic(Backend):
    name = "anthropic"

    def __init__(self, model: str = "claude-sonnet-4-5", api_key: str = "", timeout: float = 180.0, base_url: str = "https://api.anthropic.com/v1"):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def complete(self, prompt, system="", temperature=0.7, max_tokens=1024):
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            payload["system"] = system
        t0 = time.time()
        data = _post(
            f"{self.base_url}/messages",
            payload,
            {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"},
            self.timeout,
        )
        text = "".join(b.get("text", "") for b in data.get("content", []))
        return Completion(text, self.model, time.time() - t0, data.get("usage", {}))


class Ollama(Backend):
    name = "ollama"

    def __init__(self, model: str = "qwen2.5:7b", base_url: str = "http://localhost:11434", timeout: float = 300.0):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def complete(self, prompt, system="", temperature=0.7, max_tokens=1024):
        t0 = time.time()
        data = _post(
            f"{self.base_url}/api/generate",
            {
                "model": self.model,
                "prompt": prompt,
                "system": system,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            },
            {},
            self.timeout,
        )
        return Completion(data.get("response", ""), self.model, time.time() - t0)

    def embed(self, texts):
        out = []
        for t in texts:
            data = _post(f"{self.base_url}/api/embeddings", {"model": self.model, "prompt": t}, {}, self.timeout)
            out.append(data["embedding"])
        return out


class Echo(Backend):
    """Deterministic offline backend so the whole pipeline runs, and is testable, with no API key."""

    name = "echo"

    def __init__(self, model: str = "echo", handler: Callable[[str, str], str] | None = None):
        self.model = model
        self.handler = handler

    def complete(self, prompt, system="", temperature=0.7, max_tokens=1024):
        if self.handler:
            return Completion(self.handler(prompt, system), self.model)
        return Completion(json.dumps({"echo": prompt[-400:]}), self.model)

    def embed(self, texts):
        return [hashed_embedding(t) for t in texts]


def hashed_embedding(text: str, dim: int = 256) -> list[float]:
    """Deterministic bag-of-ngrams hashing embedding; no model download, decent for dedup."""
    vec = [0.0] * dim
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    grams = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
    for g in grams:
        h = int.from_bytes(hashlib.blake2b(g.encode("utf-8"), digest_size=8).digest(), "little")
        vec[h % dim] += 1.0
    norm = sum(v * v for v in vec) ** 0.5 or 1.0
    return [v / norm for v in vec]


BACKENDS = {"openai": OpenAICompatible, "vllm": OpenAICompatible, "anthropic": Anthropic, "ollama": Ollama, "echo": Echo}


def build_backend(spec: dict) -> Backend:
    kind = spec.get("backend", "echo")
    if kind not in BACKENDS:
        raise LLMError(f"unknown backend {kind!r}; choose from {sorted(BACKENDS)}")
    kwargs = {k: v for k, v in spec.items() if k != "backend"}
    return BACKENDS[kind](**kwargs)


class Runtime:
    """Role-addressed pool of backends with retry, plus an embedding fallback."""

    def __init__(self, specs: dict[str, dict], retries: int = 2, backoff: float = 2.0):
        self.backends: dict[str, Backend] = {}
        for role, spec in specs.items():
            self.backends[role] = build_backend(spec)
        self.retries = retries
        self.backoff = backoff
        self.calls: list[dict] = []

    def backend(self, role: str) -> Backend:
        return self.backends.get(role) or self.backends.get("generate") or Echo()

    def complete(self, role: str, prompt: str, system: str = "", temperature: float = 0.7, max_tokens: int = 1024) -> str:
        backend = self.backend(role)
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                out = backend.complete(prompt, system, temperature, max_tokens)
                self.calls.append({"role": role, "model": out.model, "latency": out.latency, "usage": out.usage})
                return out.text
            except LLMError as exc:
                last = exc
                if attempt < self.retries:
                    time.sleep(self.backoff * (2**attempt))
        raise LLMError(f"role={role} failed after {self.retries + 1} attempts: {last}")

    def embed(self, texts: list[str]) -> list[list[float]]:
        backend = self.backends.get("embed")
        if backend is None:
            return [hashed_embedding(t) for t in texts]
        try:
            return backend.embed(texts)
        except (LLMError, NotImplementedError):
            return [hashed_embedding(t) for t in texts]


JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def parse_json(text: str, default=None):
    """Models wrap JSON in prose and fences; recover the first valid object or array."""
    candidates = []
    m = JSON_BLOCK.search(text)
    if m:
        candidates.append(m.group(1))
    candidates.append(text)
    for start, end in (("[", "]"), ("{", "}")):
        i, j = text.find(start), text.rfind(end)
        if i != -1 and j > i:
            candidates.append(text[i : j + 1])
    for cand in candidates:
        try:
            return json.loads(cand.strip())
        except (json.JSONDecodeError, ValueError):
            continue
    return default

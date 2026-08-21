"""Model runtime: local (Ollama / vLLM / any OpenAI-compatible server) and cloud providers.

Roles let one pipeline route cheap bulk generation locally and route verification to a
stronger cloud model without any call site knowing which backend it landed on.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

ROLES = ("generate", "verify", "embed", "vision")

Vector = array
"""Embeddings are packed float32 arrays, not lists — see `select.normalize` for why.

Backends return whatever their transport gives (JSON lists for the HTTP ones); `Runtime.embed` is
the boundary that packs them, so every caller gets the same type regardless of backend. Arrays are
not JSON-serialisable: call `list(vector)` before writing one out.
"""


def as_vector(values) -> Vector:
    return values if isinstance(values, array) and values.typecode == "f" else array("f", values)


IMAGE_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".gif": "image/gif"}
MAX_IMAGE_BYTES = 4 * 1024 * 1024


def encode_image(path: str) -> tuple[str, str]:
    """Return (mime, base64) for an image on disk."""
    p = Path(path)
    mime = IMAGE_MIME.get(p.suffix.lower())
    if mime is None:
        raise LLMError(f"unsupported image type: {p.suffix}")
    data = p.read_bytes()
    if len(data) > MAX_IMAGE_BYTES:
        raise LLMError(f"{p.name} is {len(data)} bytes, over the {MAX_IMAGE_BYTES} limit")
    return mime, base64.b64encode(data).decode("ascii")


class LLMError(RuntimeError):
    """A failed model call, carrying enough to decide whether trying again could ever help."""

    def __init__(self, message: str, status: int | None = None, retry_after: float | None = None):
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after

    @property
    def retryable(self) -> bool:
        # A wrong key or a bad model name fails identically on every attempt; only the transient
        # classes are worth the backoff, and 429 in particular is worth waiting out.
        if self.status is None:
            return True
        return self.status == 429 or self.status >= 500


class BudgetExceeded(RuntimeError):
    """Raised when a run reaches its token ceiling, so the caller can stop and keep what it has."""


PERMANENT_HINTS = {
    401: "the API key was rejected",
    403: "the key is valid but not permitted to use this model",
    404: "no such model or endpoint at this base_url",
    400: "the provider rejected the request body",
}


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

    def embed(self, texts: list[str]) -> list:
        """Backends may return plain lists; `Runtime.embed` packs them into vectors."""
        raise NotImplementedError(f"{self.name} has no embedding endpoint")

    def complete_vision(self, prompt: str, images: list[str], system: str = "", temperature: float = 0.7, max_tokens: int = 1024) -> Completion:
        raise NotImplementedError(f"{self.name} has no vision endpoint")


def _post(url: str, payload: dict, headers: dict, timeout: float) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read()[:500].decode("utf-8", "replace")
        hint = PERMANENT_HINTS.get(exc.code)
        raise LLMError(
            f"{url} -> {exc.code}{f' ({hint})' if hint else ''}: {detail}",
            status=exc.code,
            retry_after=_retry_after(exc),
        ) from exc
    except urllib.error.URLError as exc:
        raise LLMError(f"{url} unreachable: {exc.reason}") from exc
    except TimeoutError as exc:
        raise LLMError(f"{url} timed out after {timeout}s") from exc


def _retry_after(exc: urllib.error.HTTPError) -> float | None:
    """Providers say when to come back; guessing an interval instead just retries into the same wall."""
    raw = (exc.headers or {}).get("Retry-After")
    try:
        return max(0.0, float(raw)) if raw else None
    except (TypeError, ValueError):
        return None


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

    def complete_vision(self, prompt, images, system="", temperature=0.7, max_tokens=1024):
        content: list[dict] = []
        for path in images:
            mime, b64 = encode_image(path)
            content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
        content.append({"type": "text", "text": prompt})
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": content}]
        t0 = time.time()
        data = _post(
            f"{self.base_url}/chat/completions",
            {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens},
            self._headers(),
            self.timeout,
        )
        return Completion(data["choices"][0]["message"]["content"], self.model, time.time() - t0, data.get("usage", {}))


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

    def complete_vision(self, prompt, images, system="", temperature=0.7, max_tokens=1024):
        content: list[dict] = []
        for path in images:
            mime, b64 = encode_image(path)
            content.append({"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}})
        content.append({"type": "text", "text": prompt})
        payload = {"model": self.model, "max_tokens": max_tokens, "temperature": temperature, "messages": [{"role": "user", "content": content}]}
        if system:
            payload["system"] = system
        t0 = time.time()
        data = _post(f"{self.base_url}/messages", payload, {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}, self.timeout)
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

    def complete_vision(self, prompt, images, system="", temperature=0.7, max_tokens=1024):
        encoded = [encode_image(p)[1] for p in images]
        t0 = time.time()
        data = _post(
            f"{self.base_url}/api/generate",
            {
                "model": self.model,
                "prompt": prompt,
                "system": system,
                "images": encoded,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            },
            {},
            self.timeout,
        )
        return Completion(data.get("response", ""), self.model, time.time() - t0)


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

    def embed(self, texts) -> list[Vector]:
        return [hashed_embedding(t) for t in texts]

    def complete_vision(self, prompt, images, system="", temperature=0.7, max_tokens=1024):
        tag = " ".join(Path(p).name for p in images)
        return self.complete(f"{prompt}\n\n[images: {tag}]", system, temperature, max_tokens)


def hashed_embedding(text: str, dim: int = 256) -> Vector:
    """Deterministic bag-of-ngrams hashing embedding; no model download, decent for dedup."""
    vec = array("f", bytes(4 * dim))
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    grams = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
    for g in grams:
        h = int.from_bytes(hashlib.blake2b(g.encode("utf-8"), digest_size=8).digest(), "little")
        vec[h % dim] += 1.0
    norm = sum(v * v for v in vec) ** 0.5 or 1.0
    return array("f", (v / norm for v in vec))


BACKENDS = {"openai": OpenAICompatible, "vllm": OpenAICompatible, "anthropic": Anthropic, "ollama": Ollama, "echo": Echo}


def build_backend(spec: dict) -> Backend:
    kind = spec.get("backend", "echo")
    if kind not in BACKENDS:
        raise LLMError(f"unknown backend {kind!r}; choose from {sorted(BACKENDS)}")
    kwargs = {k: v for k, v in spec.items() if k != "backend"}
    return BACKENDS[kind](**kwargs)


class Runtime:
    """Role-addressed pool of backends with retry, plus an embedding fallback."""

    dead_after = 5

    def __init__(self, specs: dict[str, dict], retries: int = 2, backoff: float = 2.0, budget_tokens: int | None = None):
        self.backends: dict[str, Backend] = {}
        for role, spec in specs.items():
            self.backends[role] = build_backend(spec)
        self.retries = retries
        self.backoff = backoff
        self.budget_tokens = budget_tokens
        self.calls: list[dict] = []
        self.errors: list[dict] = []
        self._lock = threading.Lock()

    def backend(self, role: str) -> Backend:
        return self.backends.get(role) or self.backends.get("generate") or Echo()

    def has_vision(self) -> bool:
        return "vision" in self.backends

    def complete(self, role: str, prompt: str, system: str = "", temperature: float = 0.7, max_tokens: int = 1024) -> str:
        return self._attempt(role, lambda b: b.complete(prompt, system, temperature, max_tokens))

    def complete_vision(self, prompt: str, images: list[str], system: str = "", temperature: float = 0.7, max_tokens: int = 1024) -> str:
        return self._attempt("vision", lambda b: b.complete_vision(images=images, prompt=prompt, system=system, temperature=temperature, max_tokens=max_tokens))

    def is_dead(self) -> bool:
        """Nothing has ever succeeded and several calls have failed: the backend is misconfigured,
        not merely flaky. Worth detecting, because retries and backoff otherwise turn a wrong API key
        into hours of sleeping before an empty dataset."""
        with self._lock:
            return not self.calls and len(self.errors) >= self.dead_after

    def _attempt(self, role: str, call: Callable[[Backend], Completion]) -> str:
        backend = self.backends.get(role) or self.backend("generate")
        if self.is_dead():
            raise LLMError(f"backend is not answering; first failure was: {self.errors[0]['error']}")
        self._check_budget()
        last: LLMError | None = None
        for attempt in range(self.retries + 1):
            try:
                out = call(backend)
                self._record(role, out)
                return out.text
            except NotImplementedError as exc:
                raise LLMError(f"{role} backend {backend.name} does not support this call") from exc
            except LLMError as exc:
                last = exc
                if attempt >= self.retries or not exc.retryable:
                    break
                time.sleep(exc.retry_after if exc.retry_after is not None else self.backoff * (2**attempt))
        n = min(attempt + 1, self.retries + 1)
        failure = LLMError(f"role={role} failed after {n} attempt{'s' if n != 1 else ''}: {last}", status=last.status if last else None)
        self._note_failure(role, backend, failure)
        raise failure

    def _record(self, role: str, out: Completion) -> None:
        with self._lock:
            self.calls.append({"role": role, "model": out.model, "latency": out.latency, "usage": out.usage})

    def _note_failure(self, role: str, backend: Backend, exc: LLMError) -> None:
        with self._lock:
            self.errors.append({"role": role, "backend": backend.name, "status": exc.status, "error": str(exc)[:400]})

    def _check_budget(self) -> None:
        if self.budget_tokens is not None and self.tokens()["total"] >= self.budget_tokens:
            raise BudgetExceeded(f"spent {self.tokens()['total']} tokens, at the {self.budget_tokens} ceiling")

    def tokens(self) -> dict[str, int]:
        """Providers disagree on field names; sum whichever pair each one reported."""
        prompt = completion = 0
        with self._lock:
            usages = [c["usage"] for c in self.calls if c["usage"]]
        for u in usages:
            prompt += int(u.get("prompt_tokens") or u.get("input_tokens") or 0)
            completion += int(u.get("completion_tokens") or u.get("output_tokens") or 0)
        return {"prompt": prompt, "completion": completion, "total": prompt + completion}

    def health(self) -> dict:
        """Calls made against calls lost, per role — the difference between a slow run and a dead backend."""
        by_role: dict[str, dict] = {}
        with self._lock:
            calls, errors = list(self.calls), list(self.errors)
        for c in calls:
            by_role.setdefault(c["role"], {"ok": 0, "failed": 0})["ok"] += 1
        for e in errors:
            by_role.setdefault(e["role"], {"ok": 0, "failed": 0})["failed"] += 1
        return {
            "calls": len(calls),
            "failures": len(errors),
            "by_role": dict(sorted(by_role.items())),
            "tokens": self.tokens(),
            "first_errors": [e["error"] for e in errors[:3]],
        }

    def embed(self, texts: list[str]) -> list[Vector]:
        """Always returns packed float32 vectors, whichever backend served the request."""
        backend = self.backends.get("embed")
        if backend is None:
            return [hashed_embedding(t) for t in texts]
        try:
            return [as_vector(v) for v in backend.embed(texts)]
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

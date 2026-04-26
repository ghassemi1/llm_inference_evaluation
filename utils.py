from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


def _percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    if p <= 0:
        return float(min(values))
    if p >= 100:
        return float(max(values))

    xs = sorted(values)
    k = (len(xs) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return float(xs[f])
    d0 = xs[f] * (c - k)
    d1 = xs[c] * (k - f)
    return float(d0 + d1)


def _fmt_s(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x * 1000.0:.1f} ms"


def _estimate_tokens(text: str) -> int:
    """Best-effort token estimate without extra deps.

    If `tiktoken` is available, uses it; otherwise falls back to a rough heuristic.
    """

    try:
        import tiktoken  # type: ignore

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return len(re.findall(r"\w+|\S", text))


def _extract_delta_text(chunk: Any) -> str:
    """Extract text from an OpenAI ChatCompletionChunk-like object."""

    try:
        choice0 = chunk.choices[0]
        delta = getattr(choice0, "delta", None)
        if delta is None:
            return ""
        return getattr(delta, "content", None) or ""
    except Exception:
        return ""


def _extract_usage_tokens(chunk: Any) -> Optional[int]:
    """Streaming usage is sometimes present only on the last chunk."""

    try:
        usage = getattr(chunk, "usage", None)
        if usage is None:
            return None
        completion = getattr(usage, "completion_tokens", None)
        if completion is None:
            return None
        return int(completion)
    except Exception:
        return None


def dataset_id(name: str) -> str:
    """Convert a HF dataset id (namespace/name) into a filesystem-friendly id."""

    return name.replace("/", "__").replace("\\", "__")


def parse_kv_env(text: str) -> Dict[str, str]:
    """Parse KEY=VALUE lines (like .env) into a dict."""

    out: Dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip()
        if (v.startswith('"') and v.endswith('"')) or (
            v.startswith("'") and v.endswith("'")
        ):
            v = v[1:-1]
        if k:
            out[k] = v
    return out

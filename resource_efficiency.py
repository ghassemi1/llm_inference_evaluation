from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import httpx
from dotenv import load_dotenv

from utils import (
    _estimate_tokens,
    _extract_delta_text,
    _extract_usage_tokens,
    dataset_id,
)


def _parse_int_list(s: str) -> List[int]:
    parts = [p.strip() for p in re.split(r"[ ,]+", s.strip()) if p.strip()]
    out: List[int] = []
    for p in parts:
        try:
            out.append(int(p))
        except Exception:
            raise argparse.ArgumentTypeError(
                f"Invalid concurrency '{p}'. Use a comma/space-separated list of integers."
            )
    if not out:
        raise argparse.ArgumentTypeError("No concurrency provided.")
    if any(x <= 0 for x in out):
        raise argparse.ArgumentTypeError("Concurrency must be > 0.")
    return out


def _prompt_from_ultrachat_row(row: Dict[str, Any]) -> Optional[str]:
    msgs = None
    for key in ("messages", "conversation", "conversations"):
        if key in row:
            msgs = row.get(key)
            break

    if isinstance(msgs, list):
        for m in msgs:
            if not isinstance(m, dict):
                continue
            role = (m.get("role") or m.get("from") or "").lower()
            content = m.get("content") or m.get("value") or m.get("text")
            if (
                role in ("user", "human")
                and isinstance(content, str)
                and content.strip()
            ):
                return content.strip()
        for m in msgs:
            if isinstance(m, dict):
                content = m.get("content") or m.get("value") or m.get("text")
                if isinstance(content, str) and content.strip():
                    return content.strip()

    for key in ("prompt", "instruction", "query", "text"):
        v = row.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()

    return None


def _cap_prompt(prompt: str, max_chars: int) -> str:
    if max_chars <= 0:
        return prompt
    if len(prompt) <= max_chars:
        return prompt
    return prompt[:max_chars]


def _mean(xs: Sequence[float]) -> Optional[float]:
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def _max(xs: Sequence[float]) -> Optional[float]:
    if not xs:
        return None
    return float(max(xs))


def _safe_div(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None or b == 0:
        return None
    return float(a / b)


def _extract_metric_name(line: str) -> Optional[str]:
    m = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)", line)
    return m.group(1) if m else None


def _parse_labels(s: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not s:
        return out

    def _unescape_prom_label(v: str) -> str:
        # Prometheus exposition label escaping: \\ \n \" (and potentially others).
        return v.replace(r"\\", "\\").replace(r"\n", "\n").replace(r"\"", '"')

    parts: List[str] = []
    buf: List[str] = []
    in_quotes = False
    escape = False
    for ch in s:
        if escape:
            buf.append(ch)
            escape = False
            continue
        if in_quotes and ch == "\\":
            buf.append(ch)
            escape = True
            continue
        if ch == '"':
            in_quotes = not in_quotes
            buf.append(ch)
            continue
        if ch == "," and not in_quotes:
            part = "".join(buf).strip()
            if part:
                parts.append(part)
            buf = []
            continue
        buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)

    for part in parts:
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            continue
        if v.startswith('"') and v.endswith('"') and len(v) >= 2:
            v = _unescape_prom_label(v[1:-1])
        out[k] = v
    return out


@dataclass
class PromSample:
    name: str
    labels: Dict[str, str]
    value: float


@dataclass
class MetricsSnapshot:
    ts: float
    samples: List[PromSample] = field(default_factory=list)

    def values(self, name_pattern: str) -> List[float]:
        rx = re.compile(name_pattern, re.IGNORECASE)
        out: List[float] = []
        for s in self.samples:
            if rx.search(s.name):
                out.append(float(s.value))
        return out

    def counter(self, patterns: Sequence[str]) -> Optional[float]:
        for pat in patterns:
            rx = re.compile(pat, re.IGNORECASE)
            vals = [s.value for s in self.samples if rx.search(s.name)]
            if vals:
                return float(sum(vals))
        return None

    def gauge_max(self, patterns: Sequence[str]) -> Optional[float]:
        vals: List[float] = []
        for pat in patterns:
            vals.extend(self.values(pat))
        return _max(vals)

    def gauge_mean(self, patterns: Sequence[str]) -> Optional[float]:
        vals: List[float] = []
        for pat in patterns:
            vals.extend(self.values(pat))
        return _mean(vals)


def parse_prometheus_text(text: str) -> MetricsSnapshot:
    def _parse_value_token(tok: str) -> Optional[float]:
        t = tok.strip()
        if not t:
            return None
        lo = t.lower()
        if lo in ("nan", "+nan", "-nan"):
            return float("nan")
        if lo in ("inf", "+inf", "infinity", "+infinity"):
            return float("inf")
        if lo in ("-inf", "-infinity"):
            return float("-inf")
        try:
            return float(t)
        except Exception:
            return None

    def _parse_sample_line(line: str) -> Optional[PromSample]:
        # Prometheus exposition format: name{labels} value [timestamp]
        # Whitespace may appear inside quoted label values.
        m = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)", line)
        if not m:
            return None
        name = m.group(1)
        i = len(name)
        label_str = ""

        if i < len(line) and line[i] == "{":
            i += 1
            start = i
            in_quotes = False
            escape = False
            while i < len(line):
                ch = line[i]
                if escape:
                    escape = False
                    i += 1
                    continue
                if in_quotes and ch == "\\":
                    escape = True
                    i += 1
                    continue
                if ch == '"':
                    in_quotes = not in_quotes
                    i += 1
                    continue
                if ch == "}" and not in_quotes:
                    label_str = line[start:i]
                    i += 1
                    break
                i += 1
            else:
                return None

        rest = line[i:].strip()
        if not rest:
            return None
        value_tok = rest.split(None, 1)[0]
        value = _parse_value_token(value_tok)
        if value is None:
            return None
        labels = _parse_labels(label_str)
        return PromSample(name=name, labels=labels, value=float(value))

    samples: List[PromSample] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        s = _parse_sample_line(line)
        if s is None:
            continue
        samples.append(s)

    return MetricsSnapshot(ts=time.time(), samples=samples)


async def fetch_metrics(
    client: httpx.AsyncClient, *, url: str, timeout_s: float
) -> MetricsSnapshot:
    r = await client.get(url, timeout=timeout_s)
    r.raise_for_status()
    return parse_prometheus_text(r.text)


async def wait_ready(
    *,
    base_url_v1: str,
    insecure: bool,
    timeout_s: float,
    poll_s: float = 1.0,
) -> float:
    """Wait until GET {base_url_v1}/models succeeds.

    Returns seconds waited.
    """

    url = base_url_v1.rstrip("/") + "/models"
    start = time.perf_counter()
    async with httpx.AsyncClient(verify=not insecure, timeout=10) as client:
        while True:
            try:
                r = await client.get(url)
                if 200 <= r.status_code < 300:
                    return float(time.perf_counter() - start)
            except Exception:
                pass

            if time.perf_counter() - start >= timeout_s:
                raise TimeoutError(
                    f"Timed out waiting for server readiness at {url} after {timeout_s:.0f}s"
                )
            await asyncio.sleep(poll_s)


def load_prompts_from_hf(
    *,
    dataset: str,
    split: str,
    num_prompts: int,
    sample_seed: int,
    prompt_max_chars: int,
    hf_streaming: bool,
    data_dir: str,
    hf_insecure: bool,
) -> List[str]:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Missing dependency 'datasets'. Install it with: pip install datasets"
        ) from e

    if num_prompts <= 0:
        raise ValueError("--num-prompts must be > 0")

    # UltraChat split aliases
    split = split.strip()
    if split == "train":
        split = "train_sft"
    elif split == "test":
        split = "test_sft"

    rng = random.Random(sample_seed)

    prompts: List[str] = []

    def _maybe_configure_hf_insecure() -> None:
        if not hf_insecure:
            return
        try:
            from huggingface_hub.utils._http import (  # type: ignore
                hf_request_event_hook,
                set_client_factory,
            )

            def _insecure_factory() -> httpx.Client:
                return httpx.Client(
                    event_hooks={"request": [hf_request_event_hook]},
                    follow_redirects=True,
                    timeout=None,
                    verify=False,
                )

            set_client_factory(_insecure_factory)
        except Exception:
            return

    data_dir_abs = os.path.abspath(data_dir)
    cache_dir = os.path.join(data_dir_abs, "hf_cache")
    local_path = os.path.join(data_dir_abs, dataset_id(dataset), split)
    if hf_streaming:
        _maybe_configure_hf_insecure()
        ds = load_dataset(dataset, split=split, streaming=True)
        try:
            ds = ds.shuffle(
                seed=sample_seed, buffer_size=min(10_000, num_prompts * 20)
            )
        except Exception:
            pass
        for row in ds:
            p = _prompt_from_ultrachat_row(row)
            if not p:
                continue
            prompts.append(_cap_prompt(p, prompt_max_chars))
            if len(prompts) >= num_prompts:
                break
        if len(prompts) < num_prompts:
            raise RuntimeError(
                f"Only collected {len(prompts)} prompts from streaming dataset; requested {num_prompts}."
            )
        return prompts

    # Prefer local/offline path if available, else download to cache_dir.
    try:
        if os.path.isdir(local_path):
            from datasets import load_from_disk  # type: ignore

            ds = load_from_disk(local_path)
        else:
            _maybe_configure_hf_insecure()
            os.makedirs(cache_dir, exist_ok=True)
            ds = load_dataset(dataset, split=split, cache_dir=cache_dir)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load dataset. Tried local path '{local_path}'. Error: {e}"
        ) from e
    n = len(ds)
    if n <= 0:
        raise RuntimeError("Dataset split is empty.")

    indices = list(range(n))
    rng.shuffle(indices)
    indices = indices[: min(num_prompts * 5, n)]

    for idx in indices:
        row = ds[int(idx)]
        p = _prompt_from_ultrachat_row(row)
        if not p:
            continue
        prompts.append(_cap_prompt(p, prompt_max_chars))
        if len(prompts) >= num_prompts:
            break

    if len(prompts) < num_prompts:
        raise RuntimeError(
            f"Only extracted {len(prompts)} prompts from dataset; requested {num_prompts}."
        )
    return prompts


@dataclass
class WorkloadStats:
    completed_requests: int
    errors: int
    tokens_out_total: int
    ttft_s_first: Optional[float]
    e2e_s_first: Optional[float]


async def run_workload(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompts: Sequence[str],
    temperature: float,
    max_tokens: int,
    duration_s: float,
    concurrency: int,
    request_timeout_s: float,
    insecure: bool,
) -> WorkloadStats:
    from openai import AsyncOpenAI

    http_client = None
    if insecure:
        http_client = httpx.AsyncClient(verify=False)

    client = AsyncOpenAI(
        base_url=base_url, api_key=api_key, http_client=http_client
    )
    stop_at = time.perf_counter() + max(0.0, duration_s)

    completed = 0
    errors = 0
    tokens_total = 0
    ttft_first: Optional[float] = None
    e2e_first: Optional[float] = None

    lock = asyncio.Lock()

    async def _one_request(prompt: str) -> Tuple[Optional[float], float, int]:
        start = time.perf_counter()
        ttft: Optional[float] = None
        first_token_t: Optional[float] = None
        parts: List[str] = []
        usage_tokens: Optional[int] = None

        stream_iter = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
        )

        async for chunk in stream_iter:
            now = time.perf_counter()

            maybe_usage = _extract_usage_tokens(chunk)
            if maybe_usage is not None:
                usage_tokens = maybe_usage

            delta = _extract_delta_text(chunk)
            if not delta:
                continue
            if first_token_t is None:
                first_token_t = now
                ttft = first_token_t - start
            parts.append(delta)

        end = time.perf_counter()
        text = "".join(parts)
        tokens_out = (
            usage_tokens
            if usage_tokens is not None
            else _estimate_tokens(text)
        )
        return ttft, end - start, int(tokens_out)

    async def _worker(worker_id: int) -> None:
        nonlocal completed, errors, tokens_total, ttft_first, e2e_first
        idx = worker_id
        while True:
            if time.perf_counter() >= stop_at:
                return
            prompt = prompts[idx % len(prompts)]
            idx += concurrency

            async def _do() -> None:
                nonlocal completed, errors, tokens_total, ttft_first, e2e_first
                try:
                    ttft, e2e, tokens_out = await _one_request(prompt)
                    async with lock:
                        completed += 1
                        tokens_total += int(tokens_out)
                        if ttft_first is None:
                            ttft_first = ttft
                            e2e_first = float(e2e)
                except Exception:
                    async with lock:
                        errors += 1

            try:
                await asyncio.wait_for(_do(), timeout=request_timeout_s)
            except asyncio.TimeoutError:
                async with lock:
                    errors += 1

    tasks = [asyncio.create_task(_worker(i)) for i in range(concurrency)]
    await asyncio.gather(*tasks)

    if http_client is not None:
        try:
            await http_client.aclose()
        except Exception:
            pass

    return WorkloadStats(
        completed_requests=int(completed),
        errors=int(errors),
        tokens_out_total=int(tokens_total),
        ttft_s_first=ttft_first,
        e2e_s_first=e2e_first,
    )


@dataclass
class EfficiencyResult:
    duration_s: float
    concurrency: int
    completed_requests: int
    errors: int
    tokens_out_total: int
    client_tps: Optional[float]
    ready_wait_s: Optional[float]
    first_request_ttft_s: Optional[float]
    first_request_e2e_s: Optional[float]

    # Process (often present on vLLM /metrics)
    rss_mean_gb: Optional[float]
    rss_peak_gb: Optional[float]
    cpu_seconds_delta: Optional[float]
    cpu_util_pct: Optional[float]

    # vLLM app metrics (optional)
    waiting_mean: Optional[float]
    running_mean: Optional[float]
    queue_time_avg_s: Optional[float]
    server_tps: Optional[float]
    kv_cache_used_gb_mean: Optional[float]
    kv_cache_util_pct_mean: Optional[float]
    kv_cache_util_pct_peak: Optional[float]

    # GPU hardware metrics via DCGM (optional)
    gpu_vram_used_gb_mean: Optional[float]
    gpu_vram_used_gb_peak: Optional[float]
    gpu_util_pct_mean: Optional[float]
    gpu_power_w_mean: Optional[float]
    gpu_power_w_peak: Optional[float]
    energy_j_delta: Optional[float]
    tokens_per_watt: Optional[float]

    # Derived “efficiency” metrics
    tokens_per_s_per_gb_vram: Optional[float]
    vram_delta_per_running_req_mb: Optional[float]

    sources: Dict[str, bool]
    skipped: List[str] = field(default_factory=list)


def _snapshot_delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    d = float(b - a)
    # Counter reset / mismatch
    if d < 0:
        return None
    return d


def _series_mean(series: Sequence[Optional[float]]) -> Optional[float]:
    xs = [float(x) for x in series if x is not None]
    return _mean(xs)


def _series_max(series: Sequence[Optional[float]]) -> Optional[float]:
    xs = [float(x) for x in series if x is not None]
    return _max(xs)


def _compute_queue_time_avg_s(
    start: MetricsSnapshot, end: MetricsSnapshot
) -> Optional[float]:
    q_sum = _snapshot_delta(
        start.counter(
            [r"queue.*time.*seconds_sum", r"request.*queue.*time.*sum"]
        ),
        end.counter(
            [r"queue.*time.*seconds_sum", r"request.*queue.*time.*sum"]
        ),
    )
    q_cnt = _snapshot_delta(
        start.counter(
            [r"queue.*time.*seconds_count", r"request.*queue.*time.*count"]
        ),
        end.counter(
            [r"queue.*time.*seconds_count", r"request.*queue.*time.*count"]
        ),
    )
    if q_sum is None or q_cnt is None or q_cnt <= 0:
        return None
    return float(q_sum / q_cnt)


def _compute_server_tps(
    start: MetricsSnapshot, end: MetricsSnapshot, wall_s: float
) -> Optional[float]:
    tok_delta = _snapshot_delta(
        start.counter(
            [
                r"token.*generated.*total",
                r"generation.*tokens.*total",
                r"completion.*tokens.*total",
                r"tokens_generated_total",
            ]
        ),
        end.counter(
            [
                r"token.*generated.*total",
                r"generation.*tokens.*total",
                r"completion.*tokens.*total",
                r"tokens_generated_total",
            ]
        ),
    )
    if tok_delta is None or wall_s <= 0:
        return None
    return float(tok_delta / wall_s)


def _compute_kv_cache(
    snaps: Sequence[MetricsSnapshot], skipped: List[str]
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    used_series: List[Optional[float]] = []
    total_series: List[Optional[float]] = []

    for s in snaps:
        used = s.counter([r"kv.*cache.*used.*bytes", r"kv_cache.*used.*bytes"])
        total = s.counter(
            [
                r"kv.*cache.*total.*bytes",
                r"kv.*cache.*capacity.*bytes",
                r"kv_cache.*total.*bytes",
            ]
        )
        used_series.append(used)
        total_series.append(total)

    used_mean = _series_mean(used_series)
    total_mean = _series_mean(total_series)

    if used_mean is None or total_mean is None or total_mean <= 0:
        skipped.append("kv_cache_usage (missing kv cache metrics)")
        return None, None, None

    util_series: List[Optional[float]] = []
    for u, t in zip(used_series, total_series):
        util_series.append(
            _safe_div(u, t) * 100.0 if u is not None and t else None
        )

    util_mean = _series_mean(util_series)
    util_peak = _series_max(util_series)
    return float(used_mean / 1e9), util_mean, util_peak


def _compute_dcgm_gpu_series(
    snaps: Sequence[MetricsSnapshot], skipped: List[str]
) -> Dict[str, Optional[float]]:
    # DCGM uses MB for FB_USED/FB_FREE in many exporters.
    vram_peak_gb: Optional[float] = None
    vram_mean_gb: Optional[float] = None
    util_mean: Optional[float] = None
    p_mean: Optional[float] = None
    p_peak: Optional[float] = None

    # For VRAM, compute per-snapshot max across GPUs, then mean/peak of that.
    vram_per_snap_max_mb: List[Optional[float]] = []
    util_per_snap: List[Optional[float]] = []
    power_per_snap: List[Optional[float]] = []

    for s in snaps:
        fb_used = s.values(r"^DCGM_FI_DEV_FB_USED$")
        if not fb_used:
            # Some exporters expose variants; keep this flexible.
            fb_used = s.values(r"^DCGM_FI_DEV_FB_USED_.*$")
        if fb_used:
            vram_per_snap_max_mb.append(float(max(fb_used)))
        else:
            vram_per_snap_max_mb.append(None)

        util = s.gauge_mean(
            [
                r"^DCGM_FI_DEV_GPU_UTIL$",
                r"^DCGM_FI_DEV_SM_UTIL$",
                r"^DCGM_FI_DEV_SM_ACTIVE$",
            ]
        )
        util_per_snap.append(util)

        p = s.gauge_mean(
            [r"^DCGM_FI_DEV_POWER_USAGE$", r"^DCGM_FI_DEV_POWER_DRAW$"]
        )
        power_per_snap.append(p)

    if any(x is not None for x in vram_per_snap_max_mb):
        vram_mean_mb = _series_mean(vram_per_snap_max_mb)
        vram_peak_mb = _series_max(vram_per_snap_max_mb)
        vram_mean_gb = (
            float(vram_mean_mb / 1024.0) if vram_mean_mb is not None else None
        )
        vram_peak_gb = (
            float(vram_peak_mb / 1024.0) if vram_peak_mb is not None else None
        )
    else:
        skipped.append("gpu_vram (DCGM_FI_DEV_FB_USED missing)")

    util_mean = _series_mean(util_per_snap)
    if util_mean is None:
        skipped.append("gpu_util (DCGM_FI_DEV_GPU_UTIL missing)")

    p_mean = _series_mean(power_per_snap)
    p_peak = _series_max(power_per_snap)
    if p_mean is None:
        skipped.append("gpu_power (DCGM_FI_DEV_POWER_USAGE missing)")

    return {
        "gpu_vram_used_gb_mean": vram_mean_gb,
        "gpu_vram_used_gb_peak": vram_peak_gb,
        "gpu_util_pct_mean": util_mean,
        "gpu_power_w_mean": p_mean,
        "gpu_power_w_peak": p_peak,
    }


def _compute_dcgm_energy_delta(
    start: MetricsSnapshot, end: MetricsSnapshot
) -> Optional[float]:
    # Energy counter is optional.
    a = start.counter([r"^DCGM_FI_DEV_ENERGY_CONSUMPTION$"])
    b = end.counter([r"^DCGM_FI_DEV_ENERGY_CONSUMPTION$"])
    d = _snapshot_delta(a, b)
    return float(d) if d is not None else None


async def main_async() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Resource efficiency benchmark for a vLLM OpenAI-compatible server (Prometheus + optional DCGM)."
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1"),
    )
    parser.add_argument(
        "--api-key", default=os.getenv("OPENAI_API_KEY", "dummy123")
    )
    parser.add_argument(
        "--model", default=os.getenv("OPENAI_MODEL", "qwen2.5-3b")
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.getenv("TEMPERATURE", "0.0")),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=int(os.getenv("MAX_TOKENS", "256")),
    )

    parser.add_argument(
        "--duration-s",
        type=float,
        default=float(os.getenv("DURATION_S", "60")),
    )
    parser.add_argument(
        "--warmup-s",
        type=float,
        default=float(os.getenv("WARMUP_S", "10")),
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of parallel workers for the workload (default: 1)",
    )
    parser.add_argument(
        "--request-timeout-s",
        type=float,
        default=float(os.getenv("REQUEST_TIMEOUT_S", "120")),
    )

    # Workload prompts
    parser.add_argument(
        "--prompt",
        default=None,
        help="If set, use this single prompt repeatedly instead of a dataset.",
    )
    parser.add_argument(
        "--dataset",
        default=os.getenv("HF_DATASET", "HuggingFaceH4/ultrachat_200k"),
    )
    parser.add_argument(
        "--split",
        default=os.getenv("HF_SPLIT", "train_sft"),
    )
    parser.add_argument(
        "--data-dir",
        default=os.getenv("DATA_DIR", "data"),
        help="Base folder for local datasets and HF cache (default: ./data)",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=min(200, int(os.getenv("NUM_PROMPTS", "500"))),
        help="How many prompts to sample from the dataset (default: min(200, NUM_PROMPTS)).",
    )
    parser.add_argument(
        "--prompt-max-chars",
        type=int,
        default=int(os.getenv("PROMPT_MAX_CHARS", "600")),
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=int(os.getenv("SAMPLE_SEED", "123")),
    )
    parser.add_argument(
        "--hf-streaming",
        action="store_true",
        help="Stream dataset without fully downloading it",
    )
    parser.add_argument(
        "--hf-insecure",
        action="store_true",
        help="Disable TLS verification for Hugging Face dataset downloads (useful behind corporate proxies).",
    )

    # Metrics endpoints
    parser.add_argument(
        "--metrics-url",
        default=os.getenv("VLLM_METRICS_URL"),
        help="vLLM Prometheus endpoint (usually /metrics)",
    )
    parser.add_argument(
        "--dcgm-url",
        default=os.getenv("DCGM_METRICS_URL"),
        help="Optional DCGM exporter endpoint (GPU util/power/VRAM), e.g. http://HOST:9400/metrics",
    )
    parser.add_argument(
        "--metrics-timeout-s",
        type=float,
        default=float(os.getenv("METRICS_TIMEOUT_S", "10")),
    )
    parser.add_argument(
        "--sample-interval-s",
        type=float,
        default=2.0,
        help="Sampling interval for metrics endpoints (seconds)",
    )
    parser.add_argument(
        "--insecure",
        dest="insecure",
        action="store_true",
        help="Disable TLS verification (base URL / metrics URLs). Default: enabled (useful for ngrok/self-signed certs).",
    )
    parser.add_argument(
        "--secure",
        dest="insecure",
        action="store_false",
        help="Enable TLS verification (override the default insecure mode).",
    )
    parser.set_defaults(insecure=True)

    # Startup/readiness timing
    parser.add_argument(
        "--wait-ready",
        action="store_true",
        help="Measure how long until GET {base_url}/models succeeds (useful if server is starting up).",
    )
    parser.add_argument(
        "--wait-ready-timeout-s",
        type=float,
        default=300.0,
    )

    parser.add_argument(
        "--out", default=None, help="Write JSON results to this path"
    )

    args = parser.parse_args()

    if args.concurrency <= 0:
        print("--concurrency must be > 0", file=sys.stderr)
        return 2
    if args.duration_s <= 0:
        print("--duration-s must be > 0", file=sys.stderr)
        return 2
    if args.sample_interval_s <= 0:
        print("--sample-interval-s must be > 0", file=sys.stderr)
        return 2

    prompts: List[str]
    if args.prompt:
        prompts = [str(args.prompt)]
    else:
        prompts = load_prompts_from_hf(
            dataset=str(args.dataset),
            split=str(args.split),
            num_prompts=int(args.num_prompts),
            sample_seed=int(args.sample_seed),
            prompt_max_chars=int(args.prompt_max_chars),
            hf_streaming=bool(args.hf_streaming),
            data_dir=str(args.data_dir),
            hf_insecure=bool(args.hf_insecure) or bool(args.insecure),
        )

    ready_wait_s: Optional[float] = None
    if args.wait_ready:
        ready_wait_s = await wait_ready(
            base_url_v1=str(args.base_url),
            insecure=bool(args.insecure),
            timeout_s=float(args.wait_ready_timeout_s),
        )

    # Warmup: a short workload run to stabilize KV cache and kernels.
    if float(args.warmup_s) > 0:
        await run_workload(
            base_url=str(args.base_url),
            api_key=str(args.api_key),
            model=str(args.model),
            prompts=prompts,
            temperature=float(args.temperature),
            max_tokens=int(args.max_tokens),
            duration_s=float(args.warmup_s),
            concurrency=int(args.concurrency),
            request_timeout_s=float(args.request_timeout_s),
            insecure=bool(args.insecure),
        )

    sources: Dict[str, bool] = {"vllm_metrics": False, "dcgm": False}
    skipped: List[str] = []

    metrics_url = str(args.metrics_url).strip() if args.metrics_url else ""
    dcgm_url = str(args.dcgm_url).strip() if args.dcgm_url else ""

    async with httpx.AsyncClient(
        verify=not bool(args.insecure)
    ) as http_client:
        vllm_snaps: List[MetricsSnapshot] = []
        dcgm_snaps: List[MetricsSnapshot] = []

        baseline_vram_gb: Optional[float] = None

        async def _try_fetch_baseline() -> None:
            nonlocal baseline_vram_gb
            if not dcgm_url:
                return
            try:
                s = await fetch_metrics(
                    http_client,
                    url=dcgm_url,
                    timeout_s=float(args.metrics_timeout_s),
                )
                # baseline = per-snapshot max across GPUs
                fb_used = s.values(r"^DCGM_FI_DEV_FB_USED$")
                if fb_used:
                    baseline_vram_gb = float(max(fb_used) / 1024.0)
                    sources["dcgm"] = True
            except Exception:
                return

        await _try_fetch_baseline()

        stop_sampling_at = time.perf_counter() + float(args.duration_s)

        # Start snapshots
        vllm_start: Optional[MetricsSnapshot] = None
        dcgm_start: Optional[MetricsSnapshot] = None

        async def _fetch_start_end(tag: str) -> None:
            nonlocal vllm_start, dcgm_start
            if metrics_url:
                try:
                    s = await fetch_metrics(
                        http_client,
                        url=metrics_url,
                        timeout_s=float(args.metrics_timeout_s),
                    )
                    vllm_snaps.append(s)
                    sources["vllm_metrics"] = True
                    if tag == "start":
                        vllm_start = s
                except Exception as e:
                    skipped.append(f"vllm_metrics_snapshot_{tag} ({e})")
            if dcgm_url:
                try:
                    s = await fetch_metrics(
                        http_client,
                        url=dcgm_url,
                        timeout_s=float(args.metrics_timeout_s),
                    )
                    dcgm_snaps.append(s)
                    sources["dcgm"] = True
                    if tag == "start":
                        dcgm_start = s
                except Exception as e:
                    skipped.append(f"dcgm_snapshot_{tag} ({e})")

        await _fetch_start_end("start")

        # Sampling task
        async def _sampler() -> None:
            while time.perf_counter() < stop_sampling_at:
                await asyncio.sleep(float(args.sample_interval_s))
                await _fetch_start_end("mid")

        sampler_task = asyncio.create_task(_sampler())

        # Workload
        workload = await run_workload(
            base_url=str(args.base_url),
            api_key=str(args.api_key),
            model=str(args.model),
            prompts=prompts,
            temperature=float(args.temperature),
            max_tokens=int(args.max_tokens),
            duration_s=float(args.duration_s),
            concurrency=int(args.concurrency),
            request_timeout_s=float(args.request_timeout_s),
            insecure=bool(args.insecure),
        )

        await sampler_task
        await _fetch_start_end("end")

    wall_s = float(args.duration_s)
    client_tps = (
        float(workload.tokens_out_total / wall_s) if wall_s > 0 else None
    )

    # Compute process metrics (from vLLM /metrics if present)
    rss_series = []
    cpu_series = []
    for s in vllm_snaps:
        rss_series.append(s.gauge_mean([r"^process_resident_memory_bytes$"]))
        cpu_series.append(s.counter([r"^process_cpu_seconds_total$"]))

    rss_mean_gb = (
        float(_series_mean(rss_series) / 1e9)
        if _series_mean(rss_series)
        else None
    )
    rss_peak_gb = (
        float(_series_max(rss_series) / 1e9)
        if _series_max(rss_series)
        else None
    )

    cpu_seconds_delta: Optional[float] = None
    cpu_util_pct: Optional[float] = None
    if cpu_series and cpu_series[0] is not None and cpu_series[-1] is not None:
        cpu_seconds_delta = _snapshot_delta(
            float(cpu_series[0]), float(cpu_series[-1])
        )
        if cpu_seconds_delta is not None and wall_s > 0:
            cores = os.cpu_count() or 1
            cpu_util_pct = float(cpu_seconds_delta / (wall_s * cores) * 100.0)
    else:
        skipped.append("cpu_util (process_cpu_seconds_total missing)")

    # vLLM queue / running / waiting
    running_series = []
    waiting_series = []
    for s in vllm_snaps:
        running_series.append(
            s.gauge_mean(
                [
                    r"running.*request",
                    r"num_.*running.*request",
                    r"requests_running",
                ]
            )
        )
        waiting_series.append(
            s.gauge_mean(
                [
                    r"waiting.*request",
                    r"queued.*request",
                    r"queue.*size",
                    r"num_.*waiting.*request",
                ]
            )
        )

    running_mean = _series_mean(running_series)
    waiting_mean = _series_mean(waiting_series)

    # Deltas that require start/end snapshots
    queue_time_avg_s: Optional[float] = None
    server_tps: Optional[float] = None
    if vllm_snaps:
        start = vllm_snaps[0]
        end = vllm_snaps[-1]
        queue_time_avg_s = _compute_queue_time_avg_s(start, end)
        server_tps = _compute_server_tps(start, end, wall_s)
    else:
        skipped.append("vllm_metrics (no snapshots)")

    # KV cache
    kv_used_gb_mean, kv_util_mean, kv_util_peak = _compute_kv_cache(
        vllm_snaps, skipped
    )

    # DCGM GPU
    gpu_vram_used_gb_mean: Optional[float] = None
    gpu_vram_used_gb_peak: Optional[float] = None
    gpu_util_pct_mean: Optional[float] = None
    gpu_power_w_mean: Optional[float] = None
    gpu_power_w_peak: Optional[float] = None
    energy_j_delta: Optional[float] = None

    if dcgm_snaps:
        g = _compute_dcgm_gpu_series(dcgm_snaps, skipped)
        gpu_vram_used_gb_mean = g.get("gpu_vram_used_gb_mean")
        gpu_vram_used_gb_peak = g.get("gpu_vram_used_gb_peak")
        gpu_util_pct_mean = g.get("gpu_util_pct_mean")
        gpu_power_w_mean = g.get("gpu_power_w_mean")
        gpu_power_w_peak = g.get("gpu_power_w_peak")
        energy_j_delta = _compute_dcgm_energy_delta(
            dcgm_snaps[0], dcgm_snaps[-1]
        )
    else:
        if dcgm_url:
            skipped.append("dcgm (no snapshots; dcgm endpoint unavailable?)")

    tokens_per_watt: Optional[float] = None
    if gpu_power_w_mean is not None and wall_s > 0:
        tokens_per_watt = _safe_div(
            float(workload.tokens_out_total), gpu_power_w_mean * wall_s
        )

    tokens_per_s_per_gb_vram: Optional[float] = None
    if (
        client_tps is not None
        and gpu_vram_used_gb_mean is not None
        and gpu_vram_used_gb_mean > 0
    ):
        tokens_per_s_per_gb_vram = float(client_tps / gpu_vram_used_gb_mean)
    else:
        if gpu_vram_used_gb_mean is None:
            skipped.append(
                "tokens_per_s_per_gb_vram (missing gpu_vram_used_gb_mean)"
            )

    # Approximate “VRAM delta per running request” if we have both baseline + running gauge.
    vram_delta_per_running_req_mb: Optional[float] = None
    if baseline_vram_gb is not None and gpu_vram_used_gb_mean is not None:
        if running_mean is not None and running_mean > 0:
            delta_mb = (gpu_vram_used_gb_mean - baseline_vram_gb) * 1024.0
            vram_delta_per_running_req_mb = float(delta_mb / running_mean)
        else:
            skipped.append("vram_delta_per_running_req (missing running_mean)")

    result = EfficiencyResult(
        duration_s=wall_s,
        concurrency=int(args.concurrency),
        completed_requests=int(workload.completed_requests),
        errors=int(workload.errors),
        tokens_out_total=int(workload.tokens_out_total),
        client_tps=client_tps,
        ready_wait_s=ready_wait_s,
        first_request_ttft_s=workload.ttft_s_first,
        first_request_e2e_s=workload.e2e_s_first,
        rss_mean_gb=rss_mean_gb,
        rss_peak_gb=rss_peak_gb,
        cpu_seconds_delta=cpu_seconds_delta,
        cpu_util_pct=cpu_util_pct,
        waiting_mean=waiting_mean,
        running_mean=running_mean,
        queue_time_avg_s=queue_time_avg_s,
        server_tps=server_tps,
        kv_cache_used_gb_mean=kv_used_gb_mean,
        kv_cache_util_pct_mean=kv_util_mean,
        kv_cache_util_pct_peak=kv_util_peak,
        gpu_vram_used_gb_mean=gpu_vram_used_gb_mean,
        gpu_vram_used_gb_peak=gpu_vram_used_gb_peak,
        gpu_util_pct_mean=gpu_util_pct_mean,
        gpu_power_w_mean=gpu_power_w_mean,
        gpu_power_w_peak=gpu_power_w_peak,
        energy_j_delta=energy_j_delta,
        tokens_per_watt=tokens_per_watt,
        tokens_per_s_per_gb_vram=tokens_per_s_per_gb_vram,
        vram_delta_per_running_req_mb=vram_delta_per_running_req_mb,
        sources=sources,
        skipped=skipped,
    )

    payload = {
        "config": {
            "base_url": args.base_url,
            "model": args.model,
            "duration_s": args.duration_s,
            "warmup_s": args.warmup_s,
            "concurrency": args.concurrency,
            "metrics_url": metrics_url or None,
            "dcgm_url": dcgm_url or None,
            "sample_interval_s": args.sample_interval_s,
            "insecure": bool(args.insecure),
        },
        "result": asdict(result),
    }

    print("=== Resource efficiency ===")
    print(f"Target:      {args.base_url}")
    print(f"Metrics URL: {metrics_url if metrics_url else '(none)'}")
    print(f"DCGM URL:    {dcgm_url if dcgm_url else '(none)'}")
    if ready_wait_s is not None:
        print(f"Ready wait:  {ready_wait_s:.2f} s")
    print(
        f"Run:         warmup={float(args.warmup_s):.0f}s  duration={wall_s:.0f}s  C={int(args.concurrency)}"
    )
    print(
        f"Requests:    ok={workload.completed_requests}  err={workload.errors}"
    )
    print(
        f"Client TPS:  {client_tps:.2f}"
        if client_tps is not None
        else "Client TPS:  n/a"
    )
    if result.server_tps is not None:
        print(f"Server TPS:  {result.server_tps:.2f}")
    if result.cpu_util_pct is not None:
        print(f"CPU util:    {result.cpu_util_pct:.1f} %")
    if result.rss_peak_gb is not None:
        print(f"RSS peak:    {result.rss_peak_gb:.2f} GB")
    if result.gpu_vram_used_gb_peak is not None:
        print(f"GPU VRAM pk: {result.gpu_vram_used_gb_peak:.2f} GB")
    if result.gpu_util_pct_mean is not None:
        print(f"GPU util:    {result.gpu_util_pct_mean:.1f} %")
    if result.gpu_power_w_mean is not None:
        print(f"GPU power:   {result.gpu_power_w_mean:.1f} W")
    if result.tokens_per_watt is not None:
        print(f"Tok/W:       {result.tokens_per_watt:.3f}")
    if result.tokens_per_s_per_gb_vram is not None:
        print(f"Tok/s/GB:    {result.tokens_per_s_per_gb_vram:.2f}")
    if result.kv_cache_util_pct_mean is not None:
        print(f"KV util:     {result.kv_cache_util_pct_mean:.1f} % (mean)")

    if result.skipped:
        # Keep it short; the JSON output (if enabled) includes full details.
        print("Skipped:    " + "; ".join(result.skipped[:6]))
        if len(result.skipped) > 6:
            print(f"           ... +{len(result.skipped) - 6} more")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote JSON:  {args.out}")

    return 0


def main() -> int:
    try:
        return asyncio.run(main_async())
    except KeyboardInterrupt:
        print("Interrupted.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

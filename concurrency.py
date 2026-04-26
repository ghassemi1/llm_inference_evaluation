from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from dotenv import load_dotenv

from utils import (
    _estimate_tokens,
    _extract_delta_text,
    _extract_usage_tokens,
    _fmt_s,
    _percentile,
    dataset_id,
)


def _fmt_rate(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.2f}"


def _parse_int_list(s: str) -> List[int]:
    parts = [p.strip() for p in re.split(r"[ ,]+", s.strip()) if p.strip()]
    out: List[int] = []
    for p in parts:
        try:
            out.append(int(p))
        except Exception:
            raise argparse.ArgumentTypeError(
                f"Invalid concurrency level '{p}'. Use a comma/space-separated list of integers."
            )
    if not out:
        raise argparse.ArgumentTypeError("No concurrency levels provided.")
    if any(x <= 0 for x in out):
        raise argparse.ArgumentTypeError("Concurrency levels must be > 0.")
    return out


def _prompt_from_ultrachat_row(row: Dict[str, Any]) -> Optional[str]:
    """Extract a user prompt string from an UltraChat row.

    The dataset schema can vary by version; this handles common shapes.
    """

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
        # Fallback: first text-like message
        for m in msgs:
            if isinstance(m, dict):
                content = m.get("content") or m.get("value") or m.get("text")
                if isinstance(content, str) and content.strip():
                    return content.strip()

    # Some variants may have a plain field
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


def load_prompts(
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

    prompts: List[str] = []
    rng = random.Random(sample_seed)

    def _maybe_configure_hf_insecure() -> None:
        if not hf_insecure:
            return
        try:
            import httpx  # type: ignore
            from huggingface_hub.utils._http import hf_request_event_hook, set_client_factory  # type: ignore

            def _insecure_factory() -> httpx.Client:
                return httpx.Client(
                    event_hooks={"request": [hf_request_event_hook]},
                    follow_redirects=True,
                    timeout=None,
                    verify=False,
                )

            set_client_factory(_insecure_factory)
        except Exception:
            # If this fails, we'll rely on default SSL behavior.
            return

    # UltraChat split aliases (common expectation is 'train')
    split = split.strip()
    if split == "train":
        split = "train_sft"
    elif split == "test":
        split = "test_sft"

    data_dir_abs = os.path.abspath(data_dir)
    cache_dir = os.path.join(data_dir_abs, "hf_cache")

    # Prefer a locally saved copy under ./data/<dataset_id>/<split>
    local_path = os.path.join(data_dir_abs, dataset_id(dataset), split)

    if hf_streaming:
        _maybe_configure_hf_insecure()
        ds = load_dataset(dataset, split=split, streaming=True)
        # Streaming shuffle is optional; keeps memory bounded.
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

    # Local/offline path if available, else download to cache_dir.
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

    # Sample indices deterministically.
    indices = list(range(n))
    rng.shuffle(indices)
    indices = indices[: min(num_prompts * 5, n)]  # oversample to skip bad rows

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
class SingleRequest:
    ok: bool
    e2e_s: Optional[float]
    ttft_s: Optional[float]
    tokens_out: Optional[int]
    error: Optional[str]


@dataclass
class LevelSummary:
    concurrency: int
    duration_s: float
    completed: int
    errors: int
    rps: Optional[float]
    tps: Optional[float]
    e2e_p50_s: Optional[float]
    e2e_p95_s: Optional[float]
    e2e_p99_s: Optional[float]
    ttft_p50_s: Optional[float]
    ttft_p95_s: Optional[float]
    ttft_p99_s: Optional[float]
    tokens_out_total: int
    metrics: Dict[str, Any]


async def _measure_one_stream(
    *,
    client: Any,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
) -> Tuple[Optional[float], float, int]:
    start = time.perf_counter()

    ttft_s: Optional[float] = None
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
            ttft_s = first_token_t - start

        parts.append(delta)

    end = time.perf_counter()
    text = "".join(parts)
    tokens_out = (
        usage_tokens if usage_tokens is not None else _estimate_tokens(text)
    )
    return ttft_s, end - start, tokens_out


async def _worker_loop(
    *,
    worker_id: int,
    stop_at: float,
    prompts: Sequence[str],
    client: Any,
    model: str,
    temperature: float,
    max_tokens: int,
    request_timeout_s: float,
    concurrency: int,
    collect: bool,
    out: List[SingleRequest],
) -> None:
    if not prompts:
        return

    idx = worker_id
    while True:
        if time.perf_counter() >= stop_at:
            return
        prompt = prompts[idx % len(prompts)]
        idx += concurrency

        async def _do() -> SingleRequest:
            try:
                ttft_s, e2e_s, tokens_out = await _measure_one_stream(
                    client=client,
                    model=model,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return SingleRequest(
                    ok=True,
                    e2e_s=e2e_s,
                    ttft_s=ttft_s,
                    tokens_out=tokens_out,
                    error=None,
                )
            except Exception as e:
                return SingleRequest(
                    ok=False,
                    e2e_s=None,
                    ttft_s=None,
                    tokens_out=None,
                    error=str(e),
                )

        try:
            res = await asyncio.wait_for(_do(), timeout=request_timeout_s)
        except asyncio.TimeoutError:
            res = SingleRequest(
                ok=False,
                e2e_s=None,
                ttft_s=None,
                tokens_out=None,
                error=f"timeout after {request_timeout_s:.0f}s",
            )

        if collect:
            out.append(res)


def _parse_prometheus_metrics(text: str) -> Dict[str, Any]:
    # Minimal parser that:
    # - aggregates unlabeled metrics by name (sum)
    # - collects histogram bucket series (by name + le)
    sums: Dict[str, float] = {}
    buckets: Dict[str, Dict[float, float]] = {}

    def _parse_labels(s: str) -> Dict[str, str]:
        # Extremely small label parser: key="value",...
        out: Dict[str, str] = {}
        if not s:
            return out
        for part in s.split(","):
            part = part.strip()
            if not part or "=" not in part:
                continue
            k, v = part.split("=", 1)
            v = v.strip()
            if v.startswith('"') and v.endswith('"') and len(v) >= 2:
                v = v[1:-1]
            out[k.strip()] = v
        return out

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # metric{labels} value [timestamp]
        try:
            metric_part, value_part = line.split(None, 1)
        except ValueError:
            continue
        try:
            value_str = value_part.split(None, 1)[0]
            value = float(value_str)
        except Exception:
            continue

        name = metric_part
        label_str = ""
        if "{" in metric_part and metric_part.endswith("}"):
            name, label_str = metric_part.split("{", 1)
            label_str = label_str[:-1]

        if name.endswith("_bucket") and label_str:
            labels = _parse_labels(label_str)
            le = labels.get("le")
            if le is not None:
                try:
                    le_f = float("inf") if le == "+Inf" else float(le)
                    buckets.setdefault(name, {})[le_f] = (
                        buckets.setdefault(name, {}).get(le_f, 0.0) + value
                    )
                except Exception:
                    pass

        sums[name] = sums.get(name, 0.0) + value

    return {"sums": sums, "buckets": buckets}


def _pick_metric(
    sums: Dict[str, float], patterns: Sequence[str]
) -> Optional[float]:
    # Choose the first metric whose name matches any pattern.
    for pat in patterns:
        rx = re.compile(pat, re.IGNORECASE)
        matches = [k for k in sums.keys() if rx.search(k)]
        if matches:
            # If multiple, sum them.
            return float(sum(sums[m] for m in matches))
    return None


def _compute_counter_delta(
    start_sums: Dict[str, float],
    end_sums: Dict[str, float],
    patterns: Sequence[str],
) -> Optional[float]:
    a = _pick_metric(start_sums, patterns)
    b = _pick_metric(end_sums, patterns)
    if a is None or b is None:
        return None
    return float(b - a)


async def _fetch_metrics_snapshot(
    *,
    url: str,
    insecure: bool,
    timeout_s: float,
) -> Dict[str, Any]:
    try:
        import httpx  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Missing dependency 'httpx'. Install it with: pip install httpx"
        ) from e

    async with httpx.AsyncClient(verify=not insecure, timeout=timeout_s) as s:
        r = await s.get(url)
        r.raise_for_status()
        parsed = _parse_prometheus_metrics(r.text)
        return {"ts": time.time(), **parsed}


def _summarize_level(
    *,
    concurrency: int,
    wall_s: float,
    results: List[SingleRequest],
    metrics_info: Dict[str, Any],
) -> LevelSummary:
    ok = [r for r in results if r.ok]
    errs = [r for r in results if not r.ok]

    e2e_s = [float(r.e2e_s) for r in ok if r.e2e_s is not None]
    ttft_s = [float(r.ttft_s) for r in ok if r.ttft_s is not None]
    tokens = [int(r.tokens_out) for r in ok if r.tokens_out is not None]
    tokens_total = int(sum(tokens))

    completed = len(ok)
    errors = len(errs)

    rps = (completed / wall_s) if wall_s > 0 else None
    tps = (tokens_total / wall_s) if wall_s > 0 else None

    return LevelSummary(
        concurrency=concurrency,
        duration_s=wall_s,
        completed=completed,
        errors=errors,
        rps=rps,
        tps=tps,
        e2e_p50_s=_percentile(e2e_s, 50),
        e2e_p95_s=_percentile(e2e_s, 95),
        e2e_p99_s=_percentile(e2e_s, 99),
        ttft_p50_s=_percentile(ttft_s, 50),
        ttft_p95_s=_percentile(ttft_s, 95),
        ttft_p99_s=_percentile(ttft_s, 99),
        tokens_out_total=tokens_total,
        metrics=metrics_info,
    )


def _detect_saturation(levels: List[LevelSummary]) -> Optional[int]:
    for i in range(1, len(levels)):
        prev = levels[i - 1]
        cur = levels[i]
        if prev.e2e_p95_s is None or cur.e2e_p95_s is None:
            continue
        if prev.rps is None or cur.rps is None:
            continue
        p95_jump = (
            cur.e2e_p95_s / prev.e2e_p95_s if prev.e2e_p95_s > 0 else None
        )
        rps_gain = (cur.rps - prev.rps) / prev.rps if prev.rps > 0 else None
        if p95_jump is not None and p95_jump >= 1.5:
            return cur.concurrency
        if (
            rps_gain is not None
            and rps_gain < 0.10
            and p95_jump is not None
            and p95_jump >= 1.2
        ):
            return cur.concurrency
    return None


async def run_level(
    *,
    concurrency: int,
    prompts: Sequence[str],
    base_url: str,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    warmup_s: float,
    duration_s: float,
    request_timeout_s: float,
    insecure: bool,
    metrics_url: Optional[str],
    metrics_timeout_s: float,
) -> LevelSummary:
    from openai import AsyncOpenAI

    http_client = None
    if insecure:
        try:
            import httpx  # type: ignore

            http_client = httpx.AsyncClient(verify=False)
        except Exception as e:
            raise RuntimeError(
                "--insecure requested, but httpx is not available. Install with: pip install httpx"
            ) from e

    client = AsyncOpenAI(
        base_url=base_url, api_key=api_key, http_client=http_client
    )

    # Warmup
    if warmup_s > 0:
        warm_stop = time.perf_counter() + warmup_s
        warm_results: List[SingleRequest] = []
        tasks = [
            asyncio.create_task(
                _worker_loop(
                    worker_id=i,
                    stop_at=warm_stop,
                    prompts=prompts,
                    client=client,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    request_timeout_s=request_timeout_s,
                    concurrency=concurrency,
                    collect=False,
                    out=warm_results,
                )
            )
            for i in range(concurrency)
        ]
        await asyncio.gather(*tasks)

    metrics_info: Dict[str, Any] = {}
    snapshots: List[Dict[str, Any]] = []

    async def _maybe_snapshot(tag: str) -> None:
        if not metrics_url:
            return
        try:
            snap = await _fetch_metrics_snapshot(
                url=metrics_url,
                insecure=insecure,
                timeout_s=metrics_timeout_s,
            )
            snap["tag"] = tag
            snapshots.append(snap)
        except Exception as e:
            metrics_info.setdefault("errors", []).append(f"{tag}: {e}")

    # Measured run
    results: List[SingleRequest] = []
    start = time.perf_counter()
    stop_at = start + max(0.0, duration_s)

    await _maybe_snapshot("start")

    tasks = [
        asyncio.create_task(
            _worker_loop(
                worker_id=i,
                stop_at=stop_at,
                prompts=prompts,
                client=client,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                request_timeout_s=request_timeout_s,
                concurrency=concurrency,
                collect=True,
                out=results,
            )
        )
        for i in range(concurrency)
    ]

    if metrics_url and duration_s >= 2:
        # Snapshot around mid-window, then around end-of-window.
        await asyncio.sleep(max(0.0, duration_s / 2.0))
        await _maybe_snapshot("mid")

        remaining = max(0.0, stop_at - time.perf_counter())
        if remaining > 0:
            await asyncio.sleep(remaining)
        await _maybe_snapshot("end")

    await asyncio.gather(*tasks)

    # Normalize throughput to the configured window, not the drain time.
    wall_s = max(1e-9, float(duration_s))

    if snapshots:
        # Compute a few common metrics if present. Names vary across vLLM versions.
        start_sums = snapshots[0].get("sums", {}) if snapshots else {}
        end_sums = snapshots[-1].get("sums", {}) if snapshots else {}

        running_vals: List[float] = []
        waiting_vals: List[float] = []
        for s in snapshots:
            sums = s.get("sums", {})
            r = _pick_metric(
                sums,
                [
                    r"running.*request",
                    r"num_.*running.*request",
                    r"requests_running",
                ],
            )
            w = _pick_metric(
                sums,
                [
                    r"waiting.*request",
                    r"queued.*request",
                    r"queue.*size",
                    r"num_.*waiting.*request",
                ],
            )
            if r is not None:
                running_vals.append(float(r))
            if w is not None:
                waiting_vals.append(float(w))

        if running_vals:
            metrics_info["running_mean"] = float(
                sum(running_vals) / len(running_vals)
            )
            metrics_info["running_max"] = float(max(running_vals))
        if waiting_vals:
            metrics_info["waiting_mean"] = float(
                sum(waiting_vals) / len(waiting_vals)
            )
            metrics_info["waiting_max"] = float(max(waiting_vals))

        # Average queue time from *_sum and *_count if present.
        q_sum = _compute_counter_delta(
            start_sums,
            end_sums,
            [r"queue.*time.*seconds_sum", r"request.*queue.*time.*sum"],
        )
        q_cnt = _compute_counter_delta(
            start_sums,
            end_sums,
            [r"queue.*time.*seconds_count", r"request.*queue.*time.*count"],
        )
        if q_sum is not None and q_cnt is not None and q_cnt > 0:
            metrics_info["queue_time_avg_s"] = float(q_sum / q_cnt)

        tok_delta = _compute_counter_delta(
            start_sums,
            end_sums,
            [
                r"token.*generated.*total",
                r"generation.*tokens.*total",
                r"completion.*tokens.*total",
            ],
        )
        if tok_delta is not None and wall_s > 0:
            metrics_info["server_tps"] = float(tok_delta / wall_s)

        metrics_info["snapshots"] = [
            {"tag": s.get("tag"), "ts": s.get("ts")} for s in snapshots
        ]

    # Close custom http client if we created one
    if http_client is not None:
        try:
            await http_client.aclose()
        except Exception:
            pass

    return _summarize_level(
        concurrency=concurrency,
        wall_s=wall_s,
        results=results,
        metrics_info=metrics_info,
    )


def _print_level(s: LevelSummary) -> None:
    print(
        f"C={s.concurrency:<3d}  ok={s.completed:<6d} err={s.errors:<4d} "
        f"RPS={_fmt_rate(s.rps):>8}  TPS={_fmt_rate(s.tps):>8}  "
        f"E2E p95={_fmt_s(s.e2e_p95_s):>10}  TTFT p95={_fmt_s(s.ttft_p95_s):>10}"
    )
    if s.metrics:
        extra = []
        if "queue_time_avg_s" in s.metrics:
            extra.append(
                f"queue_avg={_fmt_s(s.metrics.get('queue_time_avg_s'))}"
            )
        if "waiting_mean" in s.metrics:
            extra.append(f"waiting_mean={s.metrics.get('waiting_mean'):.2f}")
        if "running_mean" in s.metrics:
            extra.append(f"running_mean={s.metrics.get('running_mean'):.2f}")
        if "server_tps" in s.metrics:
            extra.append(f"server_TPS={s.metrics.get('server_tps'):.2f}")
        if extra:
            print("           " + "  ".join(extra))


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Throughput under concurrency benchmark for a vLLM OpenAI-compatible server"
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv(
            "OPENAI_BASE_URL", "http://8867-173-34-61-14.ngrok-free.app/v1"
        ),
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
        "--max-tokens", type=int, default=int(os.getenv("MAX_TOKENS", "256"))
    )

    parser.add_argument(
        "--concurrency-levels",
        type=_parse_int_list,
        default=_parse_int_list(
            os.getenv("CONCURRENCY_LEVELS", "1,2,4,8,16,32")
        ),
        help="Comma/space-separated list, e.g. 1,2,4,8,16,32",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=float(os.getenv("DURATION_S", "60")),
    )
    parser.add_argument(
        "--warmup-s", type=float, default=float(os.getenv("WARMUP_S", "10"))
    )
    parser.add_argument(
        "--request-timeout-s",
        type=float,
        default=float(os.getenv("REQUEST_TIMEOUT_S", "120")),
        help="Per-request timeout; high concurrency runs can otherwise hang",
    )

    # Prompt dataset
    parser.add_argument(
        "--dataset",
        default=os.getenv("HF_DATASET", "HuggingFaceH4/ultrachat_200k"),
    )
    parser.add_argument(
        "--split",
        default=os.getenv("HF_SPLIT", "train_sft"),
        help=(
            "Dataset split. UltraChat splits include train_sft/test_sft/train_gen/test_gen. "
            "Aliases: train->train_sft, test->test_sft."
        ),
    )
    parser.add_argument(
        "--data-dir",
        default=os.getenv("DATA_DIR", "data"),
        help="Base folder for local datasets and HF cache (default: ./data)",
    )
    parser.add_argument(
        "--num-prompts", type=int, default=int(os.getenv("NUM_PROMPTS", "500"))
    )
    parser.add_argument(
        "--prompt-max-chars",
        type=int,
        default=int(os.getenv("PROMPT_MAX_CHARS", "600")),
        help="Cap prompt size for more stable throughput",
    )
    parser.add_argument(
        "--sample-seed", type=int, default=int(os.getenv("SAMPLE_SEED", "123"))
    )
    parser.add_argument(
        "--hf-streaming",
        action="store_true",
        help="Stream dataset without fully downloading it",
    )

    # vLLM metrics
    parser.add_argument(
        "--metrics-url",
        default=os.getenv("VLLM_METRICS_URL"),
        help="Prometheus metrics endpoint, e.g. http://HOST:8000/metrics",
    )
    parser.add_argument(
        "--metrics-timeout-s",
        type=float,
        default=float(os.getenv("METRICS_TIMEOUT_S", "10")),
    )

    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS verification (base URL / metrics URL; also used for HF downloads if needed)",
    )

    parser.add_argument(
        "--out", default=None, help="Write JSON results to this path"
    )
    parser.add_argument(
        "--csv", dest="csv_path", default=None, help="Write CSV summary"
    )

    args = parser.parse_args()

    print(f"Target:  {args.base_url}")
    print(f"Model:   {args.model}")
    print(f"Levels:  {args.concurrency_levels}")
    print(
        f"Run:     warmup={args.warmup_s:.0f}s  duration={args.duration_s:.0f}s per level"
    )
    print(
        f"Prompts: {args.dataset} [{args.split}]  n={args.num_prompts}  streaming={bool(args.hf_streaming)}"
    )
    if args.metrics_url:
        print(f"Metrics: {args.metrics_url}")
    print("")

    try:
        prompts = load_prompts(
            dataset=args.dataset,
            split=args.split,
            num_prompts=args.num_prompts,
            sample_seed=args.sample_seed,
            prompt_max_chars=args.prompt_max_chars,
            hf_streaming=bool(args.hf_streaming),
            data_dir=str(args.data_dir),
            hf_insecure=bool(args.insecure),
        )
    except Exception as e:
        print(f"Failed to load prompts: {e}", file=sys.stderr)
        return 2

    summaries: List[LevelSummary] = []

    async def _run_all() -> None:
        for c in args.concurrency_levels:
            print(f"=== Concurrency {c} ===")
            s = await run_level(
                concurrency=int(c),
                prompts=prompts,
                base_url=args.base_url,
                api_key=args.api_key,
                model=args.model,
                temperature=float(args.temperature),
                max_tokens=int(args.max_tokens),
                warmup_s=float(args.warmup_s),
                duration_s=float(args.duration_s),
                request_timeout_s=float(args.request_timeout_s),
                insecure=bool(args.insecure),
                metrics_url=(
                    str(args.metrics_url) if args.metrics_url else None
                ),
                metrics_timeout_s=float(args.metrics_timeout_s),
            )
            summaries.append(s)
            _print_level(s)
            print("")

    try:
        asyncio.run(_run_all())
    except KeyboardInterrupt:
        print("Interrupted.")
        return 130

    saturation = _detect_saturation(summaries)
    if saturation is not None:
        print(f"Saturation point (heuristic): concurrency ~ {saturation}")
    else:
        print("Saturation point (heuristic): not detected")

    if args.out:
        payload = {
            "config": {
                "base_url": args.base_url,
                "model": args.model,
                "concurrency_levels": args.concurrency_levels,
                "duration_s": args.duration_s,
                "warmup_s": args.warmup_s,
                "request_timeout_s": args.request_timeout_s,
                "dataset": args.dataset,
                "split": args.split,
                "num_prompts": args.num_prompts,
                "prompt_max_chars": args.prompt_max_chars,
                "sample_seed": args.sample_seed,
                "hf_streaming": bool(args.hf_streaming),
                "metrics_url": args.metrics_url,
                "insecure": bool(args.insecure),
            },
            "saturation_concurrency": saturation,
            "levels": [asdict(s) for s in summaries],
        }
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote JSON: {args.out}")

    if args.csv_path:
        with open(args.csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "concurrency",
                    "duration_s",
                    "completed",
                    "errors",
                    "rps",
                    "tps",
                    "e2e_p50_s",
                    "e2e_p95_s",
                    "e2e_p99_s",
                    "ttft_p50_s",
                    "ttft_p95_s",
                    "ttft_p99_s",
                    "tokens_out_total",
                    "queue_time_avg_s",
                    "waiting_mean",
                    "running_mean",
                    "server_tps",
                ],
            )
            w.writeheader()
            for s in summaries:
                row = asdict(s)
                metrics = row.pop("metrics", {}) or {}
                row["queue_time_avg_s"] = metrics.get("queue_time_avg_s")
                row["waiting_mean"] = metrics.get("waiting_mean")
                row["running_mean"] = metrics.get("running_mean")
                row["server_tps"] = metrics.get("server_tps")
                w.writerow(row)
        print(f"Wrote CSV: {args.csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

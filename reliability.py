from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import os
import random
import re
import statistics
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from dotenv import load_dotenv

from utils import (
    _estimate_tokens,
    _extract_delta_text,
    _extract_usage_tokens,
    _percentile,
    dataset_id,
)

from resource_efficiency import (
    _compute_dcgm_energy_delta,
    _compute_dcgm_gpu_series,
    _compute_kv_cache,
    _compute_queue_time_avg_s,
    _compute_server_tps,
    fetch_metrics,
)


def _fmt_pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x * 100.0:.2f}%"


def _fmt_s(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x * 1000.0:.1f} ms"


def _fmt_rate(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.2f}"


def _safe_mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(statistics.fmean(values))


def _safe_stdev(values: Sequence[float]) -> Optional[float]:
    if len(values) < 2:
        return None
    return float(statistics.pstdev(values))


def _safe_cv(values: Sequence[float]) -> Optional[float]:
    mean = _safe_mean(values)
    if mean is None or mean == 0:
        return None
    stdev = _safe_stdev(values)
    if stdev is None:
        return 0.0
    return float(stdev / mean)


def _normalize_text(text: str) -> str:
    return " ".join(text.split()).strip()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _window_values(
    rows: Sequence["RequestResult"],
    start_ts: float,
    end_ts: float,
) -> List["RequestResult"]:
    return [r for r in rows if start_ts <= r.started_at < end_ts]


def _sliding_windows(
    rows: Sequence["RequestResult"],
    window_s: float,
) -> List[Tuple[float, float, List["RequestResult"]]]:
    if not rows or window_s <= 0:
        return []
    xs = sorted(rows, key=lambda r: r.started_at)
    start = xs[0].started_at
    end = max(r.ended_at for r in xs)
    out: List[Tuple[float, float, List[RequestResult]]] = []
    cursor = start
    while cursor < end:
        nxt = cursor + window_s
        out.append((cursor, nxt, _window_values(xs, cursor, nxt)))
        cursor = nxt
    return out


def _parse_int_list(s: str) -> List[int]:
    parts = [p.strip() for p in re.split(r"[ ,]+", s.strip()) if p.strip()]
    out: List[int] = []
    for p in parts:
        try:
            out.append(int(p))
        except Exception as e:
            raise argparse.ArgumentTypeError(
                f"Invalid concurrency level '{p}'."
            ) from e
    if not out:
        raise argparse.ArgumentTypeError("No concurrency levels provided.")
    if any(x <= 0 for x in out):
        raise argparse.ArgumentTypeError("Concurrency levels must be > 0.")
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
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _cap_prompt(prompt: str, max_chars: int) -> str:
    if max_chars <= 0 or len(prompt) <= max_chars:
        return prompt
    return prompt[:max_chars]


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

    split = split.strip()
    if split == "train":
        split = "train_sft"
    elif split == "test":
        split = "test_sft"

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
            return

    data_dir_abs = os.path.abspath(data_dir)
    cache_dir = os.path.join(data_dir_abs, "hf_cache")
    local_path = os.path.join(data_dir_abs, dataset_id(dataset), split)
    prompts: List[str] = []
    rng = random.Random(sample_seed)

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
            prompt = _prompt_from_ultrachat_row(row)
            if not prompt:
                continue
            prompts.append(_cap_prompt(prompt, prompt_max_chars))
            if len(prompts) >= num_prompts:
                break
        if len(prompts) < num_prompts:
            raise RuntimeError(
                f"Only collected {len(prompts)} prompts from streaming dataset; requested {num_prompts}."
            )
        return prompts

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

    indices = list(range(len(ds)))
    rng.shuffle(indices)
    indices = indices[: min(num_prompts * 5, len(indices))]
    for idx in indices:
        prompt = _prompt_from_ultrachat_row(ds[int(idx)])
        if not prompt:
            continue
        prompts.append(_cap_prompt(prompt, prompt_max_chars))
        if len(prompts) >= num_prompts:
            break
    if len(prompts) < num_prompts:
        raise RuntimeError(
            f"Only extracted {len(prompts)} prompts from dataset; requested {num_prompts}."
        )
    return prompts


def load_prompts_from_file(path: str, max_chars: int) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise RuntimeError("Prompt file must contain a JSON list.")
    prompts: List[str] = []
    for item in raw:
        if isinstance(item, str) and item.strip():
            prompts.append(_cap_prompt(item.strip(), max_chars))
            continue
        if isinstance(item, dict):
            value = (
                item.get("prompt") or item.get("content") or item.get("text")
            )
            if isinstance(value, str) and value.strip():
                prompts.append(_cap_prompt(value.strip(), max_chars))
    if not prompts:
        raise RuntimeError("Prompt file did not contain any usable prompts.")
    return prompts


@dataclass
class RequestResult:
    phase: str
    prompt_id: str
    prompt_idx: int
    worker_id: int
    concurrency: int
    started_at: float
    ended_at: float
    duration_s: Optional[float]
    ttft_s: Optional[float]
    token_events: int
    tokens_out: Optional[int]
    max_gap_s: Optional[float]
    success: bool
    timed_out: bool
    bad_output: bool
    empty_output: bool
    malformed_json: bool
    error_type: Optional[str]
    error_message: Optional[str]
    output_hash: Optional[str]
    normalized_output_hash: Optional[str]
    output_preview: str


async def _measure_one_request(
    *,
    client: Any,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
) -> Tuple[str, Optional[float], float, int, Optional[int], Optional[float]]:
    started = time.perf_counter()
    first_token_t: Optional[float] = None
    ttft_s: Optional[float] = None
    token_times: List[float] = []
    usage_tokens: Optional[int] = None
    parts: List[str] = []

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
            ttft_s = first_token_t - started
        token_times.append(now)
        parts.append(delta)

    ended = time.perf_counter()
    text = "".join(parts)
    tokens_out = (
        usage_tokens if usage_tokens is not None else _estimate_tokens(text)
    )
    max_gap_s = None
    if len(token_times) >= 2:
        gaps = [b - a for a, b in zip(token_times, token_times[1:])]
        max_gap_s = max(gaps) if gaps else None
    return (
        text,
        ttft_s,
        ended - started,
        len(token_times),
        tokens_out,
        max_gap_s,
    )


def _classify_output(text: str, expect_json: bool) -> Tuple[bool, bool, bool]:
    stripped = text.strip()
    empty_output = not stripped
    malformed_json = False
    if expect_json and stripped:
        try:
            json.loads(stripped)
        except Exception:
            malformed_json = True
    bad_output = empty_output or malformed_json
    return bad_output, empty_output, malformed_json


async def _run_single_request(
    *,
    phase: str,
    prompt: str,
    prompt_idx: int,
    worker_id: int,
    concurrency: int,
    client: Any,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout_s: float,
    expect_json: bool,
) -> RequestResult:
    started_at = time.time()
    try:
        text, ttft_s, duration_s, token_events, tokens_out, max_gap_s = (
            await asyncio.wait_for(
                _measure_one_request(
                    client=client,
                    model=model,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ),
                timeout=timeout_s,
            )
        )
        bad_output, empty_output, malformed_json = _classify_output(
            text, expect_json=expect_json
        )
        normalized = _normalize_text(text)
        return RequestResult(
            phase=phase,
            prompt_id=f"{phase}:{prompt_idx}",
            prompt_idx=prompt_idx,
            worker_id=worker_id,
            concurrency=concurrency,
            started_at=started_at,
            ended_at=time.time(),
            duration_s=duration_s,
            ttft_s=ttft_s,
            token_events=token_events,
            tokens_out=tokens_out,
            max_gap_s=max_gap_s,
            success=True,
            timed_out=False,
            bad_output=bad_output,
            empty_output=empty_output,
            malformed_json=malformed_json,
            error_type=None,
            error_message=None,
            output_hash=_sha256_text(text),
            normalized_output_hash=_sha256_text(normalized),
            output_preview=text[:160],
        )
    except asyncio.TimeoutError:
        return RequestResult(
            phase=phase,
            prompt_id=f"{phase}:{prompt_idx}",
            prompt_idx=prompt_idx,
            worker_id=worker_id,
            concurrency=concurrency,
            started_at=started_at,
            ended_at=time.time(),
            duration_s=None,
            ttft_s=None,
            token_events=0,
            tokens_out=None,
            max_gap_s=None,
            success=False,
            timed_out=True,
            bad_output=False,
            empty_output=False,
            malformed_json=False,
            error_type="timeout",
            error_message=f"timeout after {timeout_s:.2f}s",
            output_hash=None,
            normalized_output_hash=None,
            output_preview="",
        )
    except Exception as e:
        return RequestResult(
            phase=phase,
            prompt_id=f"{phase}:{prompt_idx}",
            prompt_idx=prompt_idx,
            worker_id=worker_id,
            concurrency=concurrency,
            started_at=started_at,
            ended_at=time.time(),
            duration_s=None,
            ttft_s=None,
            token_events=0,
            tokens_out=None,
            max_gap_s=None,
            success=False,
            timed_out=False,
            bad_output=False,
            empty_output=False,
            malformed_json=False,
            error_type=type(e).__name__,
            error_message=str(e),
            output_hash=None,
            normalized_output_hash=None,
            output_preview="",
        )


def _error_rate(rows: Sequence[RequestResult]) -> Optional[float]:
    if not rows:
        return None
    errors = sum(1 for row in rows if not row.success)
    return float(errors / len(rows))


def _rate(rows: Sequence[RequestResult], predicate: Any) -> Optional[float]:
    if not rows:
        return None
    hits = sum(1 for row in rows if predicate(row))
    return float(hits / len(rows))


def _tail_spike_rate(
    rows: Sequence[RequestResult], threshold_s: float
) -> Optional[float]:
    successful = [
        row for row in rows if row.success and row.duration_s is not None
    ]
    if not successful:
        return None
    spikes = sum(
        1 for row in successful if float(row.duration_s) > threshold_s
    )
    return float(spikes / len(successful))


def _throughput_windows(
    rows: Sequence[RequestResult], window_s: float
) -> List[float]:
    out: List[float] = []
    for start, end, chunk in _sliding_windows(rows, window_s):
        del start, end
        successful = [row for row in chunk if row.success]
        if chunk:
            chunk_start = min(row.started_at for row in chunk)
            chunk_end = max(row.ended_at for row in chunk)
            wall_s = max(chunk_end - chunk_start, 1e-9)
            out.append(len(successful) / wall_s)
    return out


def _drift_score(
    rows: Sequence[RequestResult], window_s: float
) -> Dict[str, Optional[float]]:
    windows = _sliding_windows(rows, window_s)
    p95s: List[float] = []
    throughputs: List[float] = []
    success_rates: List[float] = []
    for _, _, chunk in windows:
        durations = [
            float(row.duration_s)
            for row in chunk
            if row.success and row.duration_s is not None
        ]
        p95 = _percentile(durations, 95)
        if p95 is not None:
            p95s.append(float(p95))
        if chunk:
            success = sum(1 for row in chunk if row.success)
            success_rates.append(success / len(chunk))
            throughputs.append(success / window_s)

    def _relative_change(values: Sequence[float]) -> Optional[float]:
        if len(values) < 2 or values[0] == 0:
            return None
        return float((values[-1] - values[0]) / values[0])

    return {
        "p95_latency_relative_change": _relative_change(p95s),
        "throughput_relative_change": _relative_change(throughputs),
        "success_rate_relative_change": _relative_change(success_rates),
    }


def _infer_oom_or_reset(rows: Sequence[RequestResult]) -> Dict[str, Any]:
    patterns = [
        re.compile(r"out of memory", re.IGNORECASE),
        re.compile(r"cuda.*oom", re.IGNORECASE),
        re.compile(r"gpu.*reset", re.IGNORECASE),
        re.compile(r"connection reset", re.IGNORECASE),
        re.compile(r"engine.*dead", re.IGNORECASE),
    ]
    matched: List[RequestResult] = []
    for row in rows:
        message = row.error_message or ""
        if any(rx.search(message) for rx in patterns):
            matched.append(row)
    return {
        "suspected_oom_or_reset_count": len(matched),
        "suspected_oom_or_reset_rate": (
            (len(matched) / len(rows)) if rows else None
        ),
        "examples": [row.error_message for row in matched[:5]],
    }


def _determinism_summary(rows: Sequence[RequestResult]) -> Dict[str, Any]:
    by_prompt: Dict[int, List[RequestResult]] = {}
    for row in rows:
        by_prompt.setdefault(row.prompt_idx, []).append(row)

    exact_scores: List[float] = []
    normalized_scores: List[float] = []
    prompt_summaries: List[Dict[str, Any]] = []

    for prompt_idx, group in sorted(by_prompt.items()):
        successful = [row for row in group if row.success and row.output_hash]
        if not successful:
            prompt_summaries.append(
                {
                    "prompt_idx": prompt_idx,
                    "runs": len(group),
                    "exact_match_ratio": None,
                    "normalized_match_ratio": None,
                }
            )
            continue

        hashes = Counter(
            row.output_hash for row in successful if row.output_hash
        )
        norm_hashes = Counter(
            row.normalized_output_hash
            for row in successful
            if row.normalized_output_hash
        )
        exact_ratio = max(hashes.values()) / len(successful)
        normalized_ratio = max(norm_hashes.values()) / len(successful)
        exact_scores.append(exact_ratio)
        normalized_scores.append(normalized_ratio)
        prompt_summaries.append(
            {
                "prompt_idx": prompt_idx,
                "runs": len(group),
                "successes": len(successful),
                "exact_match_ratio": exact_ratio,
                "normalized_match_ratio": normalized_ratio,
            }
        )

    return {
        "prompt_count": len(by_prompt),
        "exact_match_ratio_mean": _safe_mean(exact_scores),
        "normalized_match_ratio_mean": _safe_mean(normalized_scores),
        "per_prompt": prompt_summaries,
    }


def _request_summary(
    *,
    rows: Sequence[RequestResult],
    latency_spike_threshold_s: float,
    throughput_window_s: float,
) -> Dict[str, Any]:
    durations = [
        float(row.duration_s)
        for row in rows
        if row.success and row.duration_s is not None
    ]
    ttfts = [
        float(row.ttft_s)
        for row in rows
        if row.success and row.ttft_s is not None
    ]
    tokens = [
        int(row.tokens_out)
        for row in rows
        if row.success and row.tokens_out is not None
    ]
    max_gaps = [
        float(row.max_gap_s)
        for row in rows
        if row.success and row.max_gap_s is not None
    ]
    throughputs = _throughput_windows(rows, throughput_window_s)

    summary = {
        "requests": len(rows),
        "success_rate": _rate(rows, lambda row: row.success),
        "exception_rate": _rate(
            rows, lambda row: (not row.success) and (not row.timed_out)
        ),
        "timeout_rate": _rate(rows, lambda row: row.timed_out),
        "bad_output_rate": _rate(rows, lambda row: row.bad_output),
        "empty_response_rate": _rate(rows, lambda row: row.empty_output),
        "malformed_json_rate": _rate(rows, lambda row: row.malformed_json),
        "error_rate": _error_rate(rows),
        "oom_frequency": _infer_oom_or_reset(rows),
        "latency_p50_s": _percentile(durations, 50),
        "latency_p95_s": _percentile(durations, 95),
        "latency_p99_s": _percentile(durations, 99),
        "ttft_p50_s": _percentile(ttfts, 50),
        "ttft_p95_s": _percentile(ttfts, 95),
        "ttft_p99_s": _percentile(ttfts, 99),
        "tail_latency_spike_rate": _tail_spike_rate(
            rows, latency_spike_threshold_s
        ),
        "tail_latency_spike_threshold_s": latency_spike_threshold_s,
        "throughput_rps_mean": _safe_mean(throughputs),
        "throughput_rps_stdev": _safe_stdev(throughputs),
        "throughput_rps_cv": _safe_cv(throughputs),
        "throughput_window_s": throughput_window_s,
        "tokens_out_mean": _safe_mean(tokens),
        "max_gap_p95_s": _percentile(max_gaps, 95),
    }
    return summary


def _time_slices(
    rows: Sequence[RequestResult], window_s: float
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for start, end, chunk in _sliding_windows(rows, window_s):
        durations = [
            float(row.duration_s)
            for row in chunk
            if row.success and row.duration_s is not None
        ]
        success_rate = None
        if chunk:
            success_rate = sum(1 for row in chunk if row.success) / len(chunk)
        out.append(
            {
                "start_ts": start,
                "end_ts": end,
                "requests": len(chunk),
                "success_rate": success_rate,
                "p95_latency_s": _percentile(durations, 95),
                "throughput_rps": (
                    (sum(1 for row in chunk if row.success) / window_s)
                    if window_s > 0
                    else None
                ),
                "bad_output_rate": _rate(chunk, lambda row: row.bad_output),
            }
        )
    return out


def _metrics_slice(
    snaps: Sequence[Any], start_ts: float, end_ts: float
) -> List[Any]:
    return [snap for snap in snaps if start_ts <= float(snap.ts) < end_ts]


async def _poll_metrics_endpoint(
    *,
    url: str,
    insecure: bool,
    timeout_s: float,
    sample_interval_s: float,
    stop_event: asyncio.Event,
    out: List[Any],
) -> None:
    import httpx

    async with httpx.AsyncClient(
        verify=not insecure, timeout=None
    ) as metrics_client:
        while not stop_event.is_set():
            try:
                snapshot = await fetch_metrics(
                    metrics_client,
                    url=url,
                    timeout_s=timeout_s,
                )
                out.append(snapshot)
            except Exception:
                pass
            try:
                await asyncio.wait_for(
                    stop_event.wait(), timeout=sample_interval_s
                )
            except asyncio.TimeoutError:
                continue


def _summarize_metrics(
    *,
    vllm_snaps: Sequence[Any],
    dcgm_snaps: Sequence[Any],
    wall_s: float,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "vllm_samples": len(vllm_snaps),
        "dcgm_samples": len(dcgm_snaps),
    }

    if vllm_snaps:
        start = vllm_snaps[0]
        end = vllm_snaps[-1]
        waiting_mean = _safe_mean(
            [
                float(value)
                for value in (
                    snap.gauge_mean(
                        [r"waiting", r"num_waiting", r"request.*waiting"]
                    )
                    for snap in vllm_snaps
                )
                if value is not None
            ]
        )
        waiting_peak = max(
            [
                float(value)
                for value in (
                    snap.gauge_max(
                        [r"waiting", r"num_waiting", r"request.*waiting"]
                    )
                    for snap in vllm_snaps
                )
                if value is not None
            ],
            default=None,
        )
        running_mean = _safe_mean(
            [
                float(value)
                for value in (
                    snap.gauge_mean(
                        [r"running", r"num_running", r"request.*running"]
                    )
                    for snap in vllm_snaps
                )
                if value is not None
            ]
        )
        running_peak = max(
            [
                float(value)
                for value in (
                    snap.gauge_max(
                        [r"running", r"num_running", r"request.*running"]
                    )
                    for snap in vllm_snaps
                )
                if value is not None
            ],
            default=None,
        )
        rss_peak_gb = max(
            [
                float(value) / 1e9
                for value in (
                    snap.gauge_max([r"^process_resident_memory_bytes$"])
                    for snap in vllm_snaps
                )
                if value is not None
            ],
            default=None,
        )
        summary.update(
            {
                "queue_wait_avg_s": _compute_queue_time_avg_s(start, end),
                "server_tps": _compute_server_tps(start, end, wall_s),
                "waiting_mean": waiting_mean,
                "waiting_peak": waiting_peak,
                "running_mean": running_mean,
                "running_peak": running_peak,
                "rss_peak_gb": rss_peak_gb,
            }
        )
        skipped: List[str] = []
        kv_used_mean, kv_util_mean, kv_util_peak = _compute_kv_cache(
            vllm_snaps, skipped
        )
        summary.update(
            {
                "kv_cache_used_gb_mean": kv_used_mean,
                "kv_cache_util_pct_mean": kv_util_mean,
                "kv_cache_util_pct_peak": kv_util_peak,
                "skipped": skipped,
            }
        )

    if dcgm_snaps:
        dcgm_summary = _compute_dcgm_gpu_series(dcgm_snaps, [])
        dcgm_summary["energy_j_delta"] = _compute_dcgm_energy_delta(
            dcgm_snaps[0], dcgm_snaps[-1]
        )
        summary["dcgm"] = dcgm_summary

    return summary


def _correlate_time_slices(
    *,
    slices: Sequence[Dict[str, Any]],
    vllm_snaps: Sequence[Any],
    dcgm_snaps: Sequence[Any],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in slices:
        start_ts = float(item["start_ts"])
        end_ts = float(item["end_ts"])
        vllm_chunk = _metrics_slice(vllm_snaps, start_ts, end_ts)
        dcgm_chunk = _metrics_slice(dcgm_snaps, start_ts, end_ts)
        merged = dict(item)
        if vllm_chunk:
            merged["server_tps"] = _compute_server_tps(
                vllm_chunk[0],
                vllm_chunk[-1],
                max(end_ts - start_ts, 1e-9),
            )
            merged["queue_wait_avg_s"] = _compute_queue_time_avg_s(
                vllm_chunk[0],
                vllm_chunk[-1],
            )
            waiting_series = [
                snap.gauge_mean(
                    [r"waiting", r"num_waiting", r"request.*waiting"]
                )
                for snap in vllm_chunk
            ]
            running_series = [
                snap.gauge_mean(
                    [r"running", r"num_running", r"request.*running"]
                )
                for snap in vllm_chunk
            ]
            merged["waiting_mean"] = _safe_mean(
                [float(value) for value in waiting_series if value is not None]
            )
            merged["running_mean"] = _safe_mean(
                [float(value) for value in running_series if value is not None]
            )
        if dcgm_chunk:
            dcgm_summary = _compute_dcgm_gpu_series(dcgm_chunk, [])
            merged["gpu_util_pct_mean"] = dcgm_summary.get("gpu_util_pct_mean")
            merged["gpu_power_w_mean"] = dcgm_summary.get("gpu_power_w_mean")
            merged["gpu_vram_used_gb_peak"] = dcgm_summary.get(
                "gpu_vram_used_gb_peak"
            )
        out.append(merged)
    return out


async def run_repeatability_phase(
    *,
    client: Any,
    model: str,
    prompts: Sequence[str],
    repeats: int,
    temperature: float,
    max_tokens: int,
    timeout_s: float,
    expect_json: bool,
) -> List[RequestResult]:
    results: List[RequestResult] = []
    for prompt_idx, prompt in enumerate(prompts):
        for run_idx in range(repeats):
            result = await _run_single_request(
                phase="repeatability",
                prompt=prompt,
                prompt_idx=prompt_idx,
                worker_id=run_idx,
                concurrency=1,
                client=client,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_s=timeout_s,
                expect_json=expect_json,
            )
            results.append(result)
    return results


async def run_burst_phase(
    *,
    client: Any,
    model: str,
    prompts: Sequence[str],
    concurrency_levels: Sequence[int],
    requests_per_level: int,
    temperature: float,
    max_tokens: int,
    timeout_s: float,
    expect_json: bool,
) -> List[RequestResult]:
    results: List[RequestResult] = []
    if not prompts:
        return results

    for level in concurrency_levels:
        issued = 0
        prompt_idx = 0
        while issued < requests_per_level:
            batch_size = min(level, requests_per_level - issued)
            tasks = []
            for worker_id in range(batch_size):
                idx = prompt_idx % len(prompts)
                tasks.append(
                    _run_single_request(
                        phase="burst",
                        prompt=prompts[idx],
                        prompt_idx=idx,
                        worker_id=worker_id,
                        concurrency=level,
                        client=client,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        timeout_s=timeout_s,
                        expect_json=expect_json,
                    )
                )
                prompt_idx += 1
            results.extend(await asyncio.gather(*tasks))
            issued += batch_size
    return results


async def run_soak_phase(
    *,
    client: Any,
    model: str,
    prompts: Sequence[str],
    duration_s: float,
    concurrency: int,
    temperature: float,
    max_tokens: int,
    timeout_s: float,
    expect_json: bool,
) -> List[RequestResult]:
    results: List[RequestResult] = []
    if not prompts or duration_s <= 0 or concurrency <= 0:
        return results

    stop_at = time.perf_counter() + duration_s

    async def _worker(worker_id: int) -> None:
        prompt_idx = worker_id
        while time.perf_counter() < stop_at:
            idx = prompt_idx % len(prompts)
            result = await _run_single_request(
                phase="soak",
                prompt=prompts[idx],
                prompt_idx=idx,
                worker_id=worker_id,
                concurrency=concurrency,
                client=client,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_s=timeout_s,
                expect_json=expect_json,
            )
            results.append(result)
            prompt_idx += concurrency

    await asyncio.gather(*[_worker(i) for i in range(concurrency)])
    return results


def _print_phase_summary(name: str, summary: Dict[str, Any]) -> None:
    print(f"[{name}]")
    print(f"  requests:            {summary['requests']}")
    print(f"  success rate:        {_fmt_pct(summary['success_rate'])}")
    print(f"  error rate:          {_fmt_pct(summary['error_rate'])}")
    print(f"  timeout rate:        {_fmt_pct(summary['timeout_rate'])}")
    print(f"  bad-output rate:     {_fmt_pct(summary['bad_output_rate'])}")
    print(f"  malformed JSON rate: {_fmt_pct(summary['malformed_json_rate'])}")
    print(f"  latency p95:         {_fmt_s(summary['latency_p95_s'])}")
    print(f"  latency p99:         {_fmt_s(summary['latency_p99_s'])}")
    print(f"  TTFT p95:            {_fmt_s(summary['ttft_p95_s'])}")
    print(
        f"  throughput mean:     {_fmt_rate(summary['throughput_rps_mean'])} req/s"
    )
    print(f"  throughput CV:       {_fmt_rate(summary['throughput_rps_cv'])}")
    print("")


def _write_csv(path: str, rows: Sequence[RequestResult]) -> None:
    fieldnames = (
        list(asdict(rows[0]).keys())
        if rows
        else list(RequestResult.__dataclass_fields__.keys())
    )
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


async def main_async() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Reliability and stability benchmark for a vLLM OpenAI-compatible server"
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
        "--max-tokens", type=int, default=int(os.getenv("MAX_TOKENS", "256"))
    )
    parser.add_argument(
        "--request-timeout-s",
        type=float,
        default=float(os.getenv("REQUEST_TIMEOUT_S", "60")),
    )
    parser.add_argument(
        "--repeat-prompts",
        type=int,
        default=int(os.getenv("RELIABILITY_REPEAT_PROMPTS", "5")),
    )
    parser.add_argument(
        "--repeat-runs",
        type=int,
        default=int(os.getenv("RELIABILITY_REPEAT_RUNS", "5")),
    )
    parser.add_argument(
        "--concurrency-levels",
        type=_parse_int_list,
        default=_parse_int_list(
            os.getenv("RELIABILITY_CONCURRENCY_LEVELS", "1,2,4,8")
        ),
    )
    parser.add_argument(
        "--burst-requests-per-level",
        type=int,
        default=int(os.getenv("RELIABILITY_BURST_REQUESTS_PER_LEVEL", "20")),
    )
    parser.add_argument(
        "--soak-duration-s",
        type=float,
        default=float(os.getenv("RELIABILITY_SOAK_DURATION_S", "0")),
    )
    parser.add_argument(
        "--soak-concurrency",
        type=int,
        default=int(os.getenv("RELIABILITY_SOAK_CONCURRENCY", "4")),
    )
    parser.add_argument(
        "--latency-spike-threshold-s",
        type=float,
        default=float(os.getenv("LATENCY_SPIKE_THRESHOLD_S", "5.0")),
    )
    parser.add_argument(
        "--window-s",
        type=float,
        default=float(os.getenv("RELIABILITY_WINDOW_S", "60")),
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="If set, repeat this prompt instead of loading prompts from a file or dataset.",
    )
    parser.add_argument(
        "--prompts-file",
        default=None,
        help="Path to a JSON file containing prompt strings or objects with a prompt/content/text field.",
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
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=int(os.getenv("NUM_PROMPTS", "50")),
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
    )
    parser.add_argument(
        "--hf-insecure",
        action="store_true",
    )
    parser.add_argument(
        "--expect-json",
        action="store_true",
        help="Validate that responses are valid JSON and count malformed JSON as bad output.",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS verification for the OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--metrics-url",
        default=os.getenv("VLLM_METRICS_URL"),
        help="Optional vLLM Prometheus metrics endpoint for server-side correlation.",
    )
    parser.add_argument(
        "--with-metrics",
        action="store_true",
        help="Enable vLLM Prometheus sampling using VLLM_METRICS_URL from .env unless --metrics-url is provided.",
    )
    parser.add_argument(
        "--dcgm-url",
        default=os.getenv("DCGM_METRICS_URL"),
        help="Optional DCGM exporter endpoint for GPU metrics correlation.",
    )
    parser.add_argument(
        "--with-dcgm",
        action="store_true",
        help="Enable DCGM sampling using DCGM_METRICS_URL from .env unless --dcgm-url is provided.",
    )
    parser.add_argument(
        "--metrics-timeout-s",
        type=float,
        default=float(os.getenv("METRICS_TIMEOUT_S", "10")),
    )
    parser.add_argument(
        "--sample-interval-s",
        type=float,
        default=float(os.getenv("SAMPLE_INTERVAL_S", "5")),
        help="Sampling interval for optional metrics correlation.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Write summary JSON to this path.",
    )
    parser.add_argument(
        "--csv-out",
        default=None,
        help="Write one row per request to this CSV path.",
    )
    args = parser.parse_args()

    if args.repeat_prompts <= 0:
        print("--repeat-prompts must be > 0", file=sys.stderr)
        return 2
    if args.repeat_runs <= 0:
        print("--repeat-runs must be > 0", file=sys.stderr)
        return 2
    if args.burst_requests_per_level <= 0:
        print("--burst-requests-per-level must be > 0", file=sys.stderr)
        return 2
    if args.soak_duration_s < 0:
        print("--soak-duration-s must be >= 0", file=sys.stderr)
        return 2
    if args.sample_interval_s <= 0:
        print("--sample-interval-s must be > 0", file=sys.stderr)
        return 2

    metrics_url = str(args.metrics_url).strip() if args.metrics_url else ""
    dcgm_url = str(args.dcgm_url).strip() if args.dcgm_url else ""

    if args.with_metrics and not metrics_url:
        print(
            "--with-metrics was set but VLLM_METRICS_URL / --metrics-url is missing",
            file=sys.stderr,
        )
        return 2
    if args.with_dcgm and not dcgm_url:
        print(
            "--with-dcgm was set but DCGM_METRICS_URL / --dcgm-url is missing",
            file=sys.stderr,
        )
        return 2

    if args.prompt:
        prompts = [_cap_prompt(str(args.prompt), int(args.prompt_max_chars))]
    elif args.prompts_file:
        prompts = load_prompts_from_file(
            str(args.prompts_file), int(args.prompt_max_chars)
        )
    else:
        prompts = load_prompts_from_hf(
            dataset=str(args.dataset),
            split=str(args.split),
            num_prompts=max(int(args.num_prompts), int(args.repeat_prompts)),
            sample_seed=int(args.sample_seed),
            prompt_max_chars=int(args.prompt_max_chars),
            hf_streaming=bool(args.hf_streaming),
            data_dir=str(args.data_dir),
            hf_insecure=bool(args.hf_insecure) or bool(args.insecure),
        )

    repeat_prompts = prompts[: int(args.repeat_prompts)]

    from openai import AsyncOpenAI

    http_client = None
    if args.insecure:
        try:
            import httpx

            http_client = httpx.AsyncClient(verify=False, timeout=None)
        except Exception as e:
            print(
                f"Failed to create insecure http client: {e}", file=sys.stderr
            )
            return 2

    client = AsyncOpenAI(
        base_url=str(args.base_url),
        api_key=str(args.api_key),
        http_client=http_client,
    )

    all_rows: List[RequestResult] = []
    vllm_snaps: List[Any] = []
    dcgm_snaps: List[Any] = []
    stop_event = asyncio.Event()
    metric_tasks: List[asyncio.Task[Any]] = []

    if args.with_metrics and metrics_url:
        metric_tasks.append(
            asyncio.create_task(
                _poll_metrics_endpoint(
                    url=metrics_url,
                    insecure=bool(args.insecure),
                    timeout_s=float(args.metrics_timeout_s),
                    sample_interval_s=float(args.sample_interval_s),
                    stop_event=stop_event,
                    out=vllm_snaps,
                )
            )
        )
    if args.with_dcgm and dcgm_url:
        metric_tasks.append(
            asyncio.create_task(
                _poll_metrics_endpoint(
                    url=dcgm_url,
                    insecure=bool(args.insecure),
                    timeout_s=float(args.metrics_timeout_s),
                    sample_interval_s=float(args.sample_interval_s),
                    stop_event=stop_event,
                    out=dcgm_snaps,
                )
            )
        )

    benchmark_started_at = time.time()

    try:
        repeat_rows = await run_repeatability_phase(
            client=client,
            model=str(args.model),
            prompts=repeat_prompts,
            repeats=int(args.repeat_runs),
            temperature=float(args.temperature),
            max_tokens=int(args.max_tokens),
            timeout_s=float(args.request_timeout_s),
            expect_json=bool(args.expect_json),
        )
        all_rows.extend(repeat_rows)

        burst_rows = await run_burst_phase(
            client=client,
            model=str(args.model),
            prompts=prompts,
            concurrency_levels=list(args.concurrency_levels),
            requests_per_level=int(args.burst_requests_per_level),
            temperature=float(args.temperature),
            max_tokens=int(args.max_tokens),
            timeout_s=float(args.request_timeout_s),
            expect_json=bool(args.expect_json),
        )
        all_rows.extend(burst_rows)

        soak_rows = await run_soak_phase(
            client=client,
            model=str(args.model),
            prompts=prompts,
            duration_s=float(args.soak_duration_s),
            concurrency=int(args.soak_concurrency),
            temperature=float(args.temperature),
            max_tokens=int(args.max_tokens),
            timeout_s=float(args.request_timeout_s),
            expect_json=bool(args.expect_json),
        )
        all_rows.extend(soak_rows)
    finally:
        stop_event.set()
        if metric_tasks:
            await asyncio.gather(*metric_tasks, return_exceptions=True)
        await client.close()
        if http_client is not None:
            await http_client.aclose()

    benchmark_ended_at = time.time()
    benchmark_wall_s = max(benchmark_ended_at - benchmark_started_at, 1e-9)

    phase_rows = {
        "repeatability": [
            row for row in all_rows if row.phase == "repeatability"
        ],
        "burst": [row for row in all_rows if row.phase == "burst"],
        "soak": [row for row in all_rows if row.phase == "soak"],
    }

    phase_summaries: Dict[str, Dict[str, Any]] = {}
    for phase_name, rows in phase_rows.items():
        if not rows:
            continue
        summary = _request_summary(
            rows=rows,
            latency_spike_threshold_s=float(args.latency_spike_threshold_s),
            throughput_window_s=float(args.window_s),
        )
        phase_summaries[phase_name] = summary
        _print_phase_summary(phase_name, summary)

    base_time_slices = _time_slices(all_rows, float(args.window_s))
    summary_payload: Dict[str, Any] = {
        "config": {
            "base_url": args.base_url,
            "model": args.model,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "request_timeout_s": args.request_timeout_s,
            "concurrency_levels": list(args.concurrency_levels),
            "burst_requests_per_level": args.burst_requests_per_level,
            "soak_duration_s": args.soak_duration_s,
            "soak_concurrency": args.soak_concurrency,
            "window_s": args.window_s,
            "expect_json": args.expect_json,
            "with_metrics": args.with_metrics,
            "with_dcgm": args.with_dcgm,
            "metrics_url": metrics_url if args.with_metrics else None,
            "dcgm_url": dcgm_url if args.with_dcgm else None,
            "metrics_timeout_s": args.metrics_timeout_s,
            "sample_interval_s": args.sample_interval_s,
        },
        "overall": _request_summary(
            rows=all_rows,
            latency_spike_threshold_s=float(args.latency_spike_threshold_s),
            throughput_window_s=float(args.window_s),
        ),
        "determinism": _determinism_summary(phase_rows["repeatability"]),
        "drift": _drift_score(
            phase_rows["soak"] if phase_rows["soak"] else all_rows,
            float(args.window_s),
        ),
        "time_slices": _correlate_time_slices(
            slices=base_time_slices,
            vllm_snaps=vllm_snaps,
            dcgm_snaps=dcgm_snaps,
        ),
        "server_metrics": _summarize_metrics(
            vllm_snaps=vllm_snaps,
            dcgm_snaps=dcgm_snaps,
            wall_s=benchmark_wall_s,
        ),
        "phases": phase_summaries,
        "row_count": len(all_rows),
    }

    print("[overall]")
    print(f"  requests:            {summary_payload['overall']['requests']}")
    print(
        f"  success rate:        {_fmt_pct(summary_payload['overall']['success_rate'])}"
    )
    print(
        f"  timeout rate:        {_fmt_pct(summary_payload['overall']['timeout_rate'])}"
    )
    print(
        f"  bad-output rate:     {_fmt_pct(summary_payload['overall']['bad_output_rate'])}"
    )
    print(
        f"  latency p99:         {_fmt_s(summary_payload['overall']['latency_p99_s'])}"
    )
    print(
        f"  throughput CV:       {_fmt_rate(summary_payload['overall']['throughput_rps_cv'])}"
    )
    print("")
    if summary_payload["server_metrics"].get("vllm_samples"):
        print("[server metrics]")
        print(
            f"  server TPS:          {_fmt_rate(summary_payload['server_metrics'].get('server_tps'))} tok/s"
        )
        print(
            f"  queue wait avg:      {_fmt_s(summary_payload['server_metrics'].get('queue_wait_avg_s'))}"
        )
        print(
            f"  waiting mean/peak:   {_fmt_rate(summary_payload['server_metrics'].get('waiting_mean'))} / {_fmt_rate(summary_payload['server_metrics'].get('waiting_peak'))}"
        )
        print(
            f"  running mean/peak:   {_fmt_rate(summary_payload['server_metrics'].get('running_mean'))} / {_fmt_rate(summary_payload['server_metrics'].get('running_peak'))}"
        )
        print(
            f"  RSS peak:            {_fmt_rate(summary_payload['server_metrics'].get('rss_peak_gb'))} GB"
        )
        print("")
    print("[determinism]")
    print(
        f"  exact mean:          {_fmt_pct(summary_payload['determinism']['exact_match_ratio_mean'])}"
    )
    print(
        f"  normalized mean:     {_fmt_pct(summary_payload['determinism']['normalized_match_ratio_mean'])}"
    )

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2)
            f.write("\n")

    if args.csv_out:
        _write_csv(str(args.csv_out), all_rows)

    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())

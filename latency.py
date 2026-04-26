from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from typing import Iterable, List, Optional, Tuple

from dotenv import load_dotenv

from utils import (
    _estimate_tokens,
    _extract_delta_text,
    _extract_usage_tokens,
    _fmt_s,
    _percentile,
)


def _fmt_rate(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.2f} tok/s"


@dataclass
class RequestMetrics:
    ttft_s: Optional[float]
    e2e_s: float
    token_events: int
    tokens_out: Optional[int]
    tps: Optional[float]
    itl_s: List[float]
    max_gap_s: Optional[float]
    stall_time_s: Optional[float]


def measure_one(
    *,
    client,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    stall_threshold_s: float,
    stream: bool,
) -> Tuple[RequestMetrics, str]:
    start = time.perf_counter()

    if not stream:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        end = time.perf_counter()
        text = resp.choices[0].message.content or ""
        tokens_out = None
        try:
            tokens_out = (
                int(resp.usage.completion_tokens) if resp.usage else None
            )
        except Exception:
            tokens_out = None
        if tokens_out is None:
            tokens_out = _estimate_tokens(text)
        e2e_s = end - start
        tps = (tokens_out / e2e_s) if e2e_s > 0 else None
        metrics = RequestMetrics(
            ttft_s=None,
            e2e_s=e2e_s,
            token_events=0,
            tokens_out=tokens_out,
            tps=tps,
            itl_s=[],
            max_gap_s=None,
            stall_time_s=None,
        )
        return metrics, text

    # Streaming mode
    ttft_s: Optional[float] = None
    first_token_t: Optional[float] = None
    token_times: List[float] = []
    parts: List[str] = []
    usage_tokens: Optional[int] = None

    stream_iter = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
        # If supported by server+SDK, this includes usage on the final chunk.
        stream_options={"include_usage": True},
    )

    for chunk in stream_iter:
        now = time.perf_counter()

        maybe_usage = _extract_usage_tokens(chunk)
        if maybe_usage is not None:
            usage_tokens = maybe_usage

        delta_text = _extract_delta_text(chunk)
        if not delta_text:
            continue

        if first_token_t is None:
            first_token_t = now
            ttft_s = first_token_t - start

        token_times.append(now)
        parts.append(delta_text)

    end = time.perf_counter()
    full_text = "".join(parts)

    itl_s: List[float] = []
    if len(token_times) >= 2:
        itl_s = [t2 - t1 for t1, t2 in zip(token_times, token_times[1:])]

    max_gap_s = max(itl_s) if itl_s else None
    stall_time_s = None
    if itl_s:
        stall_time_s = sum(max(0.0, gap - stall_threshold_s) for gap in itl_s)

    tokens_out = usage_tokens
    if tokens_out is None:
        tokens_out = _estimate_tokens(full_text)

    e2e_s = end - start
    gen_s = None
    if first_token_t is not None:
        gen_s = end - first_token_t

    tps = None
    if tokens_out is not None:
        denom = (
            gen_s
            if (gen_s is not None and gen_s > 0)
            else (e2e_s if e2e_s > 0 else None)
        )
        if denom:
            tps = tokens_out / denom

    metrics = RequestMetrics(
        ttft_s=ttft_s,
        e2e_s=e2e_s,
        token_events=len(token_times),
        tokens_out=tokens_out,
        tps=tps,
        itl_s=itl_s,
        max_gap_s=max_gap_s,
        stall_time_s=stall_time_s,
    )
    return metrics, full_text


def _summarize(name: str, values_s: List[float]) -> str:
    return (
        f"{name}: p50={_fmt_s(_percentile(values_s, 50))} "
        f"p95={_fmt_s(_percentile(values_s, 95))} "
        f"p99={_fmt_s(_percentile(values_s, 99))}"
    )


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Latency benchmark for a vLLM OpenAI-compatible server"
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
        "--prompt",
        default=os.getenv("PROMPT", "Explain TensorRT-LLM in 2-3 sentences."),
        help="User prompt used for each request",
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
        "--repeat", type=int, default=int(os.getenv("REPEAT", "10"))
    )
    parser.add_argument(
        "--warmup", type=int, default=int(os.getenv("WARMUP", "1"))
    )
    parser.add_argument(
        "--stall-threshold-ms",
        type=float,
        default=float(os.getenv("STALL_THRESHOLD_MS", "250")),
        help="Inter-token gap above this counts as a stall",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming (TTFT/ITL won't be available)",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS verification (only relevant for https base URLs)",
    )
    parser.add_argument(
        "--out", default=None, help="Write JSON results to this path"
    )

    args = parser.parse_args()

    # Lazy imports so argparse/help works even if deps are missing.
    from openai import OpenAI

    http_client = None
    if args.insecure:
        try:
            import httpx

            http_client = httpx.Client(verify=False)
        except Exception as e:
            print(
                f"Failed to create insecure http client (install httpx?): {e}",
                file=sys.stderr,
            )
            return 2

    client = OpenAI(
        base_url=args.base_url, api_key=args.api_key, http_client=http_client
    )

    stream = not args.no_stream
    stall_threshold_s = max(0.0, args.stall_threshold_ms / 1000.0)

    print(f"Target: {args.base_url}")
    print(f"Model:  {args.model}")
    print(f"Mode:   {'streaming' if stream else 'non-streaming'}")
    print(f"Repeat: {args.repeat} (warmup {args.warmup})")
    print("")

    all_metrics: List[RequestMetrics] = []

    total = args.warmup + args.repeat
    for i in range(total):
        is_warmup = i < args.warmup
        label = (
            f"warmup {i + 1}/{args.warmup}"
            if is_warmup
            else f"run {i + 1 - args.warmup}/{args.repeat}"
        )
        try:
            m, _ = measure_one(
                client=client,
                model=args.model,
                prompt=args.prompt,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                stall_threshold_s=stall_threshold_s,
                stream=stream,
            )
        except Exception as e:
            print(f"{label}: ERROR: {e}", file=sys.stderr)
            continue

        if is_warmup:
            print(f"{label}: e2e={_fmt_s(m.e2e_s)} (ignored)")
            continue

        all_metrics.append(m)
        print(
            f"{label}: ttft={_fmt_s(m.ttft_s)} e2e={_fmt_s(m.e2e_s)} "
            f"tps={_fmt_rate(m.tps)} max_gap={_fmt_s(m.max_gap_s)}"
        )

    if not all_metrics:
        print("No successful runs to summarize.", file=sys.stderr)
        return 1

    e2e_s = [m.e2e_s for m in all_metrics]
    ttft_s = [m.ttft_s for m in all_metrics if m.ttft_s is not None]
    tps = [m.tps for m in all_metrics if m.tps is not None]
    max_gaps = [m.max_gap_s for m in all_metrics if m.max_gap_s is not None]
    stall_times = [
        m.stall_time_s for m in all_metrics if m.stall_time_s is not None
    ]

    # ITL distribution (flattened across requests)
    itl_all: List[float] = []
    for m in all_metrics:
        itl_all.extend(m.itl_s)

    ttft_p50 = _percentile(ttft_s, 50) if ttft_s else None
    ttft_p95 = _percentile(ttft_s, 95) if ttft_s else None
    ttft_p99 = _percentile(ttft_s, 99) if ttft_s else None
    e2e_p50 = _percentile(e2e_s, 50)
    e2e_p95 = _percentile(e2e_s, 95)
    e2e_p99 = _percentile(e2e_s, 99)
    itl_p50 = _percentile(itl_all, 50) if itl_all else None
    itl_p95 = _percentile(itl_all, 95) if itl_all else None
    itl_p99 = _percentile(itl_all, 99) if itl_all else None
    itl_mean = statistics.mean(itl_all) if itl_all else None
    max_gap_p95 = _percentile(max_gaps, 95) if max_gaps else None
    max_gap_p99 = _percentile(max_gaps, 99) if max_gaps else None
    stall_time_p95 = _percentile(stall_times, 95) if stall_times else None
    stall_time_p99 = _percentile(stall_times, 99) if stall_times else None
    tps_p50 = _percentile(tps, 50) if tps else None
    tps_p95 = _percentile(tps, 95) if tps else None
    tps_p99 = _percentile(tps, 99) if tps else None
    token_events = [m.token_events for m in all_metrics]
    tokens_out = [
        m.tokens_out for m in all_metrics if m.tokens_out is not None
    ]
    token_events_mean = statistics.mean(token_events)
    completion_tokens_mean = (
        statistics.mean([float(x) for x in tokens_out]) if tokens_out else None
    )

    print("\nSummary")
    print("-" * 60)
    if ttft_s:
        print(
            f"TTFT: p50={_fmt_s(ttft_p50)} p95={_fmt_s(ttft_p95)} p99={_fmt_s(ttft_p99)}"
        )
    else:
        print("TTFT: n/a (run with streaming to measure TTFT)")
    print(
        f"End-to-end: p50={_fmt_s(e2e_p50)} p95={_fmt_s(e2e_p95)} p99={_fmt_s(e2e_p99)}"
    )

    if itl_all:
        print(
            f"Inter-token latency (ITL): p50={_fmt_s(itl_p50)} p95={_fmt_s(itl_p95)} p99={_fmt_s(itl_p99)}"
        )
        print(f"ITL mean: {_fmt_s(itl_mean)}")
    else:
        print("Inter-token latency (ITL): n/a (no token events captured)")

    if max_gaps:
        print(
            f"Max gap (per request) p95={_fmt_s(max_gap_p95)} p99={_fmt_s(max_gap_p99)}"
        )
    if stall_times:
        print(
            "Stall time (sum of per-gap excess over "
            f"{args.stall_threshold_ms:.0f}ms): p95={_fmt_s(stall_time_p95)} "
            f"p99={_fmt_s(stall_time_p99)}"
        )

    if tps:
        print(
            f"Tokens/sec: p50={_fmt_rate(tps_p50)} "
            f"p95={_fmt_rate(tps_p95)} "
            f"p99={_fmt_rate(tps_p99)}"
        )
    else:
        print("Tokens/sec: n/a")

    # Small sanity details
    print(
        f"Token events (stream chunks w/ text): mean={token_events_mean:.1f}"
    )
    if tokens_out:
        print(f"Completion tokens: mean={completion_tokens_mean:.1f}")

    if args.out:
        payload = {
            "config": {
                "base_url": args.base_url,
                "model": args.model,
                "prompt": args.prompt,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "repeat": args.repeat,
                "warmup": args.warmup,
                "stall_threshold_ms": args.stall_threshold_ms,
                "stream": stream,
                "insecure": bool(args.insecure),
            },
            "summary": {
                "sample_count": len(all_metrics),
                "ttft_p50_s": ttft_p50,
                "ttft_p95_s": ttft_p95,
                "ttft_p99_s": ttft_p99,
                "e2e_p50_s": e2e_p50,
                "e2e_p95_s": e2e_p95,
                "e2e_p99_s": e2e_p99,
                "itl_p50_s": itl_p50,
                "itl_p95_s": itl_p95,
                "itl_p99_s": itl_p99,
                "itl_mean_s": itl_mean,
                "max_gap_p95_s": max_gap_p95,
                "max_gap_p99_s": max_gap_p99,
                "stall_time_p95_s": stall_time_p95,
                "stall_time_p99_s": stall_time_p99,
                "tokens_per_second_p50": tps_p50,
                "tokens_per_second_p95": tps_p95,
                "tokens_per_second_p99": tps_p99,
                "token_events_mean": token_events_mean,
                "completion_tokens_mean": completion_tokens_mean,
            },
            "requests": [asdict(m) for m in all_metrics],
        }
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote JSON: {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

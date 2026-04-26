from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import httpx
from dotenv import load_dotenv


def _preview(text: str, *, max_lines: int) -> str:
    lines = text.splitlines()
    head = lines[:max_lines]
    more = len(lines) - len(head)
    out = "\n".join(head)
    if more > 0:
        out += f"\n... ({more} more lines)"
    return out


def _guess_metrics_url_from_base_url(base_url: str) -> Optional[str]:
    # Common case: OpenAI-compatible base URL is http(s)://host:port/v1
    # Metrics is usually http(s)://host:port/metrics
    base_url = base_url.strip().rstrip("/")
    if not base_url:
        return None
    if base_url.endswith("/v1"):
        return base_url[: -len("/v1")] + "/metrics"
    return None


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Sanity-check vLLM Prometheus metrics endpoint (/metrics)."
    )
    parser.add_argument(
        "--metrics-url",
        default=os.getenv("VLLM_METRICS_URL"),
        help="Metrics endpoint, e.g. http://HOST:8000/metrics",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=float(os.getenv("METRICS_TIMEOUT_S", "10")),
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS verification (useful for self-signed HTTPS).",
    )
    parser.add_argument(
        "--preview-lines",
        type=int,
        default=30,
        help="How many lines of the response to print.",
    )

    args = parser.parse_args()

    metrics_url = (args.metrics_url or "").strip()
    if not metrics_url:
        base_url = os.getenv("OPENAI_BASE_URL", "").strip()
        guessed = _guess_metrics_url_from_base_url(base_url)
        hint = (
            f"\nHint: set VLLM_METRICS_URL, e.g. {guessed}"
            if guessed
            else "\nHint: set VLLM_METRICS_URL to http(s)://HOST:PORT/metrics"
        )
        print(
            "Missing --metrics-url / VLLM_METRICS_URL." + hint, file=sys.stderr
        )
        return 2

    print(f"Metrics URL: {metrics_url}")

    try:
        with httpx.Client(
            verify=not args.insecure, timeout=args.timeout_s
        ) as client:
            r = client.get(metrics_url)
            r.raise_for_status()
    except Exception as e:
        print(f"Failed to fetch metrics: {e}", file=sys.stderr)
        return 1

    text = r.text or ""
    if "# HELP" not in text and "# TYPE" not in text:
        print(
            "Warning: response does not look like Prometheus text format.",
            file=sys.stderr,
        )

    print(f"HTTP {r.status_code} ({len(text)} bytes)")
    print("")
    print(_preview(text, max_lines=max(1, int(args.preview_lines))))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

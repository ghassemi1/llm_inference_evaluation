# LLM Inference Benchmarks (vLLM + OpenAI-compatible API)

This repo contains small, focused scripts to benchmark a vLLM-served model (OpenAI-compatible `/v1` API) for:

- **Single-request latency** (TTFT, end-to-end, inter-token gaps) via `latency.py`
- **Throughput under concurrency** (RPS/TPS, tail latency, saturation) via `concurrency.py`
- **Reliability and stability** (success/error/timeout/bad-output rates, determinism, throughput variance, drift) via `reliability.py`
- **Answer quality on summarization datasets** (ROUGE, ROUGE-Lsum, chrF, optional BLEU/BLEURT/BARTScore, extractiveness, repetition, heuristic entity faithfulness, and API-side perplexity probe) via `quality_eval.py`

It’s designed for *serving performance* (not evaluation frameworks, RAG, or multi-agent workflows).

`quality_eval.py` adds a lightweight offline evaluation path for the model itself. It does not add any RAG or agent evaluation layer.

## Setup

### 1) Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 2) Configure `.env`

Create/edit `.env` (a template is included in this repo) and set at least:

```env
OPENAI_BASE_URL=http://XXXXXX/v1
OPENAI_API_KEY=XXXX
OPENAI_MODEL=qwen2.5-3b
```

Optional (recommended for `concurrency.py`) — vLLM Prometheus metrics:

```env
# Usually the same host/port as OPENAI_BASE_URL, but without /v1
VLLM_METRICS_URL=http://XXXXXXXXX/metrics
```

All scripts call `dotenv.load_dotenv()` and then use environment variables as defaults.

Optional for Hugging Face-backed metrics such as BERTScore, BLEURT, or model downloads:

```env
HF_TOKEN=YOUR_HUGGINGFACE_TOKEN
```

## Dataset (UltraChat prompts)

The concurrency benchmark can draw prompts from a HuggingFace dataset. This repo uses:

- `HF_DATASET=HuggingFaceH4/ultrachat_200k`
- Split: `train_sft` (alias: `train`)

### Download a small local subset into `./data`

This is recommended to keep the benchmark stable and avoid large downloads.

```bash
python download.py \
  --dataset HuggingFaceH4/ultrachat_200k \
  --split train \
  --data-dir data \
  --max-rows 1000 \
  --streaming \
  --insecure
```

Notes:
- `--streaming` + `--max-rows` materializes only N rows and usually keeps disk usage small.
- Use `--insecure` only if your network causes `CERTIFICATE_VERIFY_FAILED` when accessing HuggingFace.

`concurrency.py` prefers loading from `data/HuggingFaceH4__ultrachat_200k/<split>` if present.

## Reliability and stability benchmark

`reliability.py` is a black-box benchmark for the served endpoint rather than a model-quality evaluator. It focuses on connection and serving stability for your OpenAI-compatible vLLM deployment.

If you run `python reliability.py` with no extra flags, it uses the built-in defaults:

- repeatability phase: `--repeat-prompts 5` and `--repeat-runs 5`
- burst phase: `--concurrency-levels 1,2,4,8` and `--burst-requests-per-level 20`
- soak phase: disabled by default because `--soak-duration-s 0`

So a plain default run is a small dataset-based repeatability + burst benchmark, not a 30-minute or 60-minute soak test.

It measures:

- request success rate
- exception/crash rate
- timeout rate
- bad-output rate
- empty-response rate
- malformed-JSON rate (when `--expect-json` is enabled)
- latency p50/p95/p99 and TTFT p50/p95/p99
- tail-latency spike rate above a configurable threshold
- throughput variance across time windows
- determinism / repeatability for the same prompt
- drift over time across rolling windows
- suspected OOM / GPU-reset frequency based on error text patterns

Recommended prompt sources:

- **Repeatability / determinism**: use a fixed local prompt or a small JSON prompts file.
- **Burst / soak load**: use `HuggingFaceH4/ultrachat_200k` prompts, preferably from the cached local subset under `./data`.
- **Malformed JSON checks**: use prompts that explicitly demand strict JSON and run with `--expect-json`.

### Minimal smoke test

```bash
python reliability.py \
  --prompt "what is llm?" \
  --repeat-prompts 1 \
  --repeat-runs 2 \
  --concurrency-levels 1,2 \
  --burst-requests-per-level 4 \
  --soak-duration-s 0
```

### Determinism / repeatability test

Use temperature `0` and a fixed prompt set.

```bash
python reliability.py \
  --prompts-file prompts_repeatability.json \
  --repeat-prompts 10 \
  --repeat-runs 10 \
  --concurrency-levels 1 \
  --burst-requests-per-level 10 \
  --soak-duration-s 0 \
  --temperature 0 \
  --out reliability_repeatability.json \
  --csv-out reliability_repeatability.csv
```

### Burst stability test

This is the main check for success rate, timeout rate, tail spikes, and throughput variance under load.

```bash
python reliability.py \
  --dataset HuggingFaceH4/ultrachat_200k \
  --split train \
  --num-prompts 100 \
  --concurrency-levels 1,2,4,8,16 \
  --burst-requests-per-level 40 \
  --request-timeout-s 60 \
  --window-s 60 \
  --out reliability_burst.json \
  --csv-out reliability_burst.csv
```

### Soak / drift test

Use this for long-duration stability and drift-over-time checks.

Start with a short confidence run first. If that stays clean, increase the soak duration to `1800` or `3600` seconds.

#### Short confidence soak

```bash
python reliability.py \
  --dataset HuggingFaceH4/ultrachat_200k \
  --split train \
  --num-prompts 60 \
  --concurrency-levels 1,2,4 \
  --burst-requests-per-level 12 \
  --soak-duration-s 300 \
  --soak-concurrency 2 \
  --window-s 60 \
  --with-metrics \
  --with-dcgm \
  --out reliability_soak_short.json \
  --csv-out reliability_soak_short.csv
```

#### Longer soak after confidence run

```bash
python reliability.py \
  --dataset HuggingFaceH4/ultrachat_200k \
  --split train \
  --num-prompts 100 \
  --concurrency-levels 1,2,4 \
  --burst-requests-per-level 20 \
  --soak-duration-s 3600 \
  --soak-concurrency 4 \
  --window-s 300 \
  --with-metrics \
  --with-dcgm \
  --out reliability_soak.json \
  --csv-out reliability_soak.csv
```

### JSON-format reliability test

```bash
python reliability.py \
  --prompts-file prompts_json.json \
  --repeat-prompts 10 \
  --repeat-runs 5 \
  --concurrency-levels 1,2,4 \
  --burst-requests-per-level 20 \
  --expect-json \
  --with-metrics \
  --with-dcgm \
  --out reliability_json.json
```

### `reliability.py` metrics

`reliability.py` reports metrics at five levels:

1. phase summaries (`repeatability`, `burst`, `soak`)
2. an `overall` summary across all executed phases
3. `determinism` metrics from the repeatability phase
4. `drift` and `time_slices` across rolling windows
5. optional `server_metrics` correlated from vLLM Prometheus and DCGM

Below are the main metric definitions.

- **Requests**
  - Definition: total number of request attempts included in that summary block.
  - In `overall`, this is the total across all executed phases.

- **Success rate** (`success_rate`)
  - Definition: `successful_requests / total_requests`.
  - A request counts as successful when the call completes without exception or timeout.

- **Exception rate** (`exception_rate`)
  - Definition: `requests_with_non_timeout_exceptions / total_requests`.
  - This excludes explicit client-side timeouts.

- **Timeout rate** (`timeout_rate`)
  - Definition: `timed_out_requests / total_requests`.
  - A timeout is triggered by `--request-timeout-s`.

- **Error rate** (`error_rate`)
  - Definition: `failed_requests / total_requests`.
  - This includes both exception failures and timeouts.

- **Bad-output rate** (`bad_output_rate`)
  - Definition: `bad_output_requests / total_requests`.
  - A response is marked as bad output when it is empty, malformed JSON under `--expect-json`, or both.

- **Empty-response rate** (`empty_response_rate`)
  - Definition: `empty_output_requests / total_requests`.
  - Counts responses whose final text is empty after stripping whitespace.

- **Malformed-JSON rate** (`malformed_json_rate`)
  - Definition: `json_parse_failures / total_requests`.
  - Only meaningful when `--expect-json` is enabled.

- **Suspected OOM / reset frequency** (`oom_frequency`)
  - Heuristic derived from error text matching patterns such as `out of memory`, `cuda ... oom`, `gpu reset`, `connection reset`, or `engine dead`.
  - Reported as both count and rate.
  - This is not a definitive root-cause signal unless you also correlate it with server-side logs or metrics.

- **Latency p50 / p95 / p99** (`latency_p50_s`, `latency_p95_s`, `latency_p99_s`)
  - End-to-end request latency percentiles in seconds.
  - Measured only over successful requests.

- **TTFT p50 / p95 / p99** (`ttft_p50_s`, `ttft_p95_s`, `ttft_p99_s`)
  - Time-to-first-token percentiles in seconds.
  - Measured only over successful streaming requests.

- **Tail-latency spike rate** (`tail_latency_spike_rate`)
  - Definition: fraction of successful requests whose end-to-end latency exceeds `--latency-spike-threshold-s`.
  - Useful for spotting rare but operationally painful outliers.

- **Tokens-out mean** (`tokens_out_mean`)
  - Mean completion-token count per successful request.
  - Uses server-reported `usage.completion_tokens` when available, otherwise token estimation.

- **Max-gap p95** (`max_gap_p95_s`)
  - p95 of the largest inter-token gap seen within each successful request.
  - Useful for identifying bursty or stalled streaming behavior.

- **Throughput mean / stdev / CV** (`throughput_rps_mean`, `throughput_rps_stdev`, `throughput_rps_cv`)
  - These are computed from rolling request-throughput windows.
  - `throughput_rps_mean`: average requests/sec across windows.
  - `throughput_rps_stdev`: standard deviation of requests/sec across windows.
  - `throughput_rps_cv`: coefficient of variation, i.e. `stdev / mean`.
  - Higher CV means more unstable throughput.

- **Determinism exact-match ratio mean** (`exact_match_ratio_mean`)
  - For each repeated prompt, compute the fraction of successful runs that match the most common exact output byte-for-byte.
  - Then average that value across prompts.

- **Determinism normalized-match ratio mean** (`normalized_match_ratio_mean`)
  - Same idea as exact match, but after whitespace normalization.
  - This is usually the more useful operational repeatability metric.

- **Per-prompt determinism** (`per_prompt`)
  - Detailed breakdown for each repeated prompt: run count, success count, exact-match ratio, and normalized-match ratio.

- **Drift metrics** (`p95_latency_relative_change`, `throughput_relative_change`, `success_rate_relative_change`)
  - Compare the first and last rolling windows of the evaluated time period.
  - Example: `p95_latency_relative_change = 0.20` means the last window's p95 latency is 20% higher than the first window's p95 latency.

- **Time slices** (`time_slices`)
  - Rolling windows over the full run, using `--window-s`.
  - Each slice includes request count, success rate, p95 latency, throughput, and bad-output rate.
  - If server metrics are enabled, slices also include correlated server-side fields such as `server_tps`, `queue_wait_avg_s`, `waiting_mean`, `running_mean`, and optional GPU metrics.

- **Server TPS** (`server_tps`)
  - Optional, from vLLM Prometheus counters.
  - Definition: `delta(server_generated_token_counter) / wall_time`.
  - Useful as an internal cross-check against client-side throughput.

- **Queue wait average** (`queue_wait_avg_s`)
  - Optional, from vLLM queue-time counters.
  - Definition: `delta(queue_time_seconds_sum) / delta(queue_time_seconds_count)`.
  - Interprets how long requests spent waiting in the server queue on average.

- **Waiting / running mean and peak** (`waiting_mean`, `waiting_peak`, `running_mean`, `running_peak`)
  - Optional, from vLLM gauges.
  - Reflect the average and peak queued requests and active running requests during the benchmark window.

- **RSS peak** (`rss_peak_gb`)
  - Optional, from `process_resident_memory_bytes` on the vLLM metrics endpoint.
  - Peak resident memory used by the server process during the run.

- **KV cache metrics** (`kv_cache_used_gb_mean`, `kv_cache_util_pct_mean`, `kv_cache_util_pct_peak`)
  - Optional, from vLLM KV-cache byte gauges/counters when exposed by your vLLM version.
  - If unavailable, these remain missing and are listed under `skipped`.

- **GPU metrics from DCGM** (`gpu_vram_used_gb_mean`, `gpu_vram_used_gb_peak`, `gpu_util_pct_mean`, `gpu_power_w_mean`, `gpu_power_w_peak`, `energy_j_delta`)
  - Optional, from the DCGM exporter.
  - Useful for correlating latency or failures with VRAM pressure, GPU saturation, and power draw.

Notes:

- `reliability.py` infers suspected OOM or GPU reset events from returned error text. If you need definitive attribution, pair this with server-side Prometheus/DCGM metrics or logs.
- `--with-metrics` enables vLLM Prometheus sampling using `VLLM_METRICS_URL` from `.env`. You can still override the URL explicitly with `--metrics-url`.
- `--with-dcgm` enables GPU-side correlation using `DCGM_METRICS_URL` from `.env`. You can still override the URL explicitly with `--dcgm-url`.
- The script writes phase summaries to stdout and can also emit full JSON summaries plus per-request CSV rows.
- For a stable offline workflow, materialize only a small prompt subset under `./data` with `download.py` and reuse it across runs.

## Quality evaluation datasets

`quality_eval.py` is built to materialize only small subsets under `./data` and keep Hugging Face cache files there as well.

- Summarization datasets:
  - `cnn_dailymail` (default)
  - `xsum` (supported for later comparison)
- Perplexity dataset:
  - `wikitext` (`wikitext-2-raw-v1`)

The first run streams only a limited number of rows, saves that subset under `data/<dataset>/<split>`, and reuses it later. This avoids downloading the full dataset or using the default global HF cache directory.

### Summarization quality + perplexity probe

```bash
python quality_eval.py \
  --dataset cnn_dailymail \
  --split validation \
  --quality-samples 25 \
  --subset-rows 50 \
  --include-bleu \
  --perplexity-samples 20 \
  --perplexity-subset-rows 40 \
  --data-dir data \
  --insecure \
  --out quality_eval.json
```

### Enable optional BLEU, BLEURT, and BARTScore

`quality_eval.py` computes the default summarization metrics automatically, but these heavier metrics are opt-in:

- `--include-bleu`
- `--include-bleurt`
- `--include-bartscore`

Example with all optional metrics enabled:

```bash
python quality_eval.py \
  --dataset cnn_dailymail \
  --split validation \
  --quality-samples 25 \
  --subset-rows 50 \
  --include-bleu \
  --include-bleurt \
  --include-bartscore \
  --perplexity-samples 20 \
  --perplexity-subset-rows 40 \
  --data-dir data \
  --insecure \
  --out quality_eval.json
```

Notes:
- If `--include-bleu` is omitted, `bleu` stays `null` in the JSON output.
- If `--include-bleurt` is omitted, `bleurt` stays `null` in the JSON output.
- If `--include-bartscore` is omitted, `bartscore` stays `null` in the JSON output.
- `BARTScore` is much heavier than BLEU and BLEURT because it loads a seq2seq model for scoring.

What this does:
- generates summaries through your OpenAI-compatible vLLM endpoint,
- scores them with ROUGE, ROUGE-Lsum, chrF, extractiveness, repetition, compression ratio, and heuristic entity-faithfulness metrics,
- optionally computes BERTScore, BLEU, BLEURT, and a source-to-summary BARTScore-style metric,
- probes whether `/v1/completions` returns prompt logprobs needed for API-side perplexity,
- if prompt logprobs are supported, measures perplexity on a small WikiText subset saved under `./data`.

### `quality_eval.py` metrics

Below are the definitions for the metrics reported by `quality_eval.py`.

- **ROUGE-1 / ROUGE-2 / ROUGE-L**
  - Reference-overlap metrics between the generated summary and the gold summary.
  - `ROUGE-1`: unigram overlap.
  - `ROUGE-2`: bigram overlap.
  - `ROUGE-L`: longest-common-subsequence overlap.

- **ROUGE-Lsum**
  - Summary-level longest-common-subsequence overlap.
  - Intended for multi-sentence summarization and usually more informative than plain `ROUGE-L` for paragraph summaries.

- **BERTScore**
  - Semantic similarity between prediction and reference using contextual embeddings.
  - Useful when wording differs but meaning is similar.

- **BLEU**
  - N-gram precision-style overlap metric.
  - More common for translation, but optionally useful here as an additional lexical signal.

- **chrF**
  - Character n-gram F-score between prediction and reference.
  - Often more robust than BLEU when wording varies slightly.

- **BLEURT**
  - Learned reference-based quality metric.
  - Uses a pretrained BLEURT model to score how well the generated summary matches the reference summary.

- **BARTScore**
  - Model-based scoring metric using a seq2seq model.
  - In this repo it is used as a source-to-summary scoring signal.

- **Compression ratio**
  - Definition: `summary_token_count / source_token_count`.
  - Lower values mean the summary is shorter relative to the source.

- **Extractive coverage**
  - Fraction of summary tokens that are part of copied fragments found in the source.
  - Higher values indicate a more extractive summary.

- **Extractive density**
  - Measures how concentrated copied fragments are in the summary.
  - Higher values indicate longer copied spans rather than many short copied pieces.

- **Novel 1-gram ratio / Novel 2-gram ratio**
  - Fraction of summary unigrams or bigrams that do not appear in the source.
  - Higher values indicate a more abstractive summary.

- **Repeated 2-gram ratio / Repeated 3-gram ratio**
  - Fraction of repeated bigrams or trigrams in the generated summary.
  - Higher values can indicate repetition or degenerate generation.

- **Entity precision**
  - Fraction of extracted summary entities that also appear in the source.
  - Higher values suggest fewer unsupported named entities in the summary.

- **Entity recall**
  - Fraction of extracted source entities that also appear in the summary.
  - This is a rough measure of how much source entity content is retained.

- **Unsupported entity rate**
  - Fraction of extracted summary entities that do not appear in the source.
  - Higher values can indicate hallucinated or unsupported entities.

- **Perplexity**
  - Computed separately from summarization overlap metrics using held-out text, currently `wikitext`.
  - Lower perplexity means the model assigns higher likelihood to the evaluation text.

- **Perplexity capability probe**
  - Checks whether the served vLLM OpenAI-compatible endpoint returns prompt logprobs through `/v1/completions`.
  - If unsupported, API-side perplexity cannot be measured by this script.

### Switch to XSum later

```bash
python quality_eval.py \
  --dataset xsum \
  --split validation \
  --quality-samples 25 \
  --subset-rows 50 \
  --data-dir data
```

### Notes

- `cnn_dailymail` is the default because ROUGE is usually easier to interpret there than on `xsum`.
- BERTScore is slower and heavier than ROUGE; use `--skip-bertscore` for a lighter run.
- If your network has certificate issues when downloading Hugging Face models or metrics, pass `--insecure`. This matters not only for datasets, but also for BERTScore, BLEURT, and other Hugging Face-backed metric assets.
- `bleu`, `bleurt`, and `bartscore` stay `null` unless you explicitly enable them with `--include-bleu`, `--include-bleurt`, or `--include-bartscore`.
- `--include-bleurt` uses the installed BLEURT package. By default it uses the bundled checkpoint; pass `--bleurt-checkpoint <local_dir>` only if you have a local exported BLEURT checkpoint directory.
- `--include-bartscore` uses a seq2seq model, default `facebook/bart-large-cnn`, to compute a source-to-summary BARTScore-style metric and is significantly heavier than the lexical metrics.
- Entity faithfulness in `quality_eval.py` is heuristic and based on named-entity-like string overlap between source and summary, not a full factuality verifier.
- API-side perplexity depends on whether the served endpoint returns prompt token logprobs. If the probe says it is unsupported, perplexity must be measured locally from the model checkpoint instead of through the API.

## Concurrency benchmark (`concurrency.py`)

### Short sanity run

A quick check that:
- prompts load correctly,
- the model endpoint is reachable,
- metrics print as expected.

```bash
python concurrency.py \
  --concurrency-levels 1,2,4 \
  --warmup-s 5 \
  --duration-s 15 \
  --data-dir data \
  --dataset HuggingFaceH4/ultrachat_200k \
  --split train
```

### Full run

```bash
python concurrency.py \
  --concurrency-levels 1,2,4,8,16,32 \
  --warmup-s 10 \
  --duration-s 60 \
  --out concurrency_results.json \
  --csv concurrency_results.csv
```

Optional vLLM metrics (Prometheus):

```bash
python concurrency.py \
  --metrics-url http://YOUR_VLLM_HOST:8000/metrics
```

### vLLM `/metrics` (Prometheus)

`VLLM_METRICS_URL` should point at the vLLM Prometheus text endpoint (usually `GET /metrics`).

Common mapping:
- If your OpenAI-compatible API base is `http(s)://HOST:PORT/v1`
- Then metrics is usually `http(s)://HOST:PORT/metrics`

#### How to test

1) Quick fetch (PowerShell note: use `curl.exe` to avoid `Invoke-WebRequest` prompts):

```bash
curl http://HOST:PORT/metrics
```

2) Using this repo’s helper script (reads `.env`):

```bash
python metrics_check.py
```

3) End-to-end test (metrics snapshots + actual LLM load):

```bash
python concurrency.py \
  --concurrency-levels 1 \
  --warmup-s 0 \
  --duration-s 10 \
  --num-prompts 5 \
  --hf-streaming
```

#### What’s the difference between Prometheus metrics vs benchmark results?

This repo produces *two kinds of numbers*:

1) **Client-side benchmark results** (from `concurrency.py` and `latency.py`)
   - These are measured by the benchmark script (your client machine) by timing requests and counting tokens in responses.
   - They answer: “What did *I* experience from outside the server?”
   - Examples:
     - `TTFT p95`: how long the client waited for the first streamed token at p95.
     - `E2E p95`: how long the client waited for the full response at p95.
     - `RPS` / `TPS`: how many requests/tokens the client completed per second.
   - These include everything between client and server: network latency, ngrok/proxy overhead, timeouts/retries, and server queueing.

2) **Server-side Prometheus metrics** (from vLLM `/metrics`)
   - These are internal counters/gauges exported by the vLLM process (and its Python/uvicorn runtime).
   - They answer: “What is happening *inside* the server while it is serving requests?”
   - Prometheus output is *not a single number* — it’s a large list of time-series. A few examples you might see:
     - Process/runtime metrics (always present in many Python services):
       - `process_resident_memory_bytes`: RAM used by the server process.
       - `process_cpu_seconds_total`: CPU time consumed.
     - vLLM serving metrics (names vary by vLLM version/config):
       - queued/waiting request gauges (how many requests are waiting vs running),
       - token counters (how many tokens have been generated),
       - queue time counters (how much time requests spent waiting in the queue).

How they relate:
- **Benchmarks tell you “outside view” performance** (what your users feel).
- **Prometheus tells you “inside view” causes** (queue buildup, running requests, server TPS, memory pressure).

Concrete example interpretation:
- If `concurrency.py` shows `TTFT p95` rising sharply at high concurrency, and Prometheus shows waiting/queue size rising at the same time, that’s strong evidence the server is saturating and requests are queueing.
- If client `E2E` is high but server queue metrics look low, the bottleneck might be network/ngrok or client-side timeouts rather than server saturation.

#### Using ngrok

If you created your tunnel with `ngrok http 8000` (same port as the vLLM OpenAI server), ngrok will typically forward both:
- `https://<your-ngrok-domain>/v1` (OpenAI-compatible API)
- `https://<your-ngrok-domain>/metrics` (Prometheus metrics)

You only need a separate tunnel if metrics is served on a different port, or if your proxy/tunnel is configured to only forward `/v1` paths.

#### GPU metrics (VRAM / utilization / power)

vLLM `/metrics` usually does **not** include GPU utilization % or power (watts). For those, run an NVIDIA Prometheus exporter such as **DCGM exporter** on the vLLM host and expose it over HTTP.

- DCGM exporter local URL is typically `http://HOST:9400/metrics`
- If you need it remotely, create a second ngrok tunnel for port `9400` and set `DCGM_METRICS_URL` to the ngrok `/metrics` URL.

## Resource Efficiency benchmark (`resource_efficiency.py`)

`resource_efficiency.py` runs an inference workload for a fixed duration and (optionally) samples:

- vLLM Prometheus metrics (`VLLM_METRICS_URL`, usually `/metrics`) for server-side TPS + process CPU/RSS
- NVIDIA GPU metrics via DCGM exporter (`DCGM_METRICS_URL`, usually `:9400/metrics`) for VRAM/util/power

It then prints a single concise summary, and can also write a JSON payload via `--out`.

### How to run

1) Minimal run (single prompt repeated):

```bash
python resource_efficiency.py \
  --duration-s 10 \
  --warmup-s 0 \
  --concurrency 1 \
  --prompt "Explain LLM inference in 3 sentences."
```

2) Include GPU metrics (DCGM exporter) and/or HTTPS endpoints:

```bash
python resource_efficiency.py \
  --duration-s 30 \
  --warmup-s 5 \
  --concurrency 2 \
  --prompt "Write a short paragraph about caching." \
  --insecure
```

Notes:
- By default, `resource_efficiency.py` runs in **insecure** mode (TLS verification disabled) to work smoothly with ngrok/self-signed certs.
- Use `--secure` if you want certificate verification enforced.
- To override endpoints without editing `.env`, pass `--metrics-url` and/or `--dcgm-url`.
- If you omit `--prompt`, the script samples prompts from the configured dataset (see the Dataset section). For fully offline runs, prefer downloading a local subset into `./data` first.

Environment variables used as defaults:

```env
# OpenAI-compatible endpoint
OPENAI_BASE_URL=http://HOST:8000/v1
OPENAI_API_KEY=...
OPENAI_MODEL=...

# vLLM Prometheus metrics (optional but recommended)
VLLM_METRICS_URL=http://HOST:8000/metrics

# DCGM exporter GPU metrics (optional)
DCGM_METRICS_URL=http://HOST:9400/metrics
```

### `resource_efficiency.py` metrics

`resource_efficiency.py` reports a mix of client-measured throughput and server/hardware telemetry.

- **Requests (ok/err)**
  - `ok`: completed requests during the run window.
  - `err`: exceptions/timeouts.

- **Client TPS**
  - Definition: `tokens_out_total / duration_s` measured by the benchmark client.
  - Tokens are taken from the OpenAI streaming `usage.completion_tokens` when available, else estimated.

- **Server TPS** (optional, from vLLM `/metrics`)
  - Definition: `delta(server_token_counter) / duration_s`.
  - Useful as a cross-check against `Client TPS`.

- **CPU util** (optional, from vLLM `/metrics`)
  - Approx definition: `delta(process_cpu_seconds_total) / (duration_s * cpu_cores) * 100`.

- **RSS peak** (optional, from vLLM `/metrics`)
  - Peak of `process_resident_memory_bytes` over the sampling window.

- **GPU VRAM pk** (optional, from DCGM)
  - Peak of `DCGM_FI_DEV_FB_USED` (per-snapshot max across GPUs), converted from MB → GB.

- **GPU util** (optional, from DCGM)
  - Mean of `DCGM_FI_DEV_GPU_UTIL` over the sampling window.

- **GPU power** (optional, from DCGM)
  - Mean of `DCGM_FI_DEV_POWER_USAGE` over the sampling window (watts).

- **Tok/W** (optional; derived)
  - Definition: `tokens_out_total / (gpu_power_w_mean * duration_s)`.
  - Interpretation: tokens generated per joule (since watts × seconds = joules).

- **Tok/s/GB** (optional; derived)
  - Definition: `Client TPS / gpu_vram_used_gb_mean`.
  - Interpretation: token throughput per GB of VRAM used (a rough “VRAM efficiency” proxy).

- **KV util** (optional, from vLLM `/metrics`)
  - KV cache utilization percent, derived from KV used/total byte counters or gauges.
  - If you see `Skipped: kv_cache_usage (missing kv cache metrics)`, it usually means your vLLM `/metrics` does not expose KV cache byte metrics (names vary by version/config), so the script can’t compute it.

- **Skipped**
  - A short list of metrics the script could not compute (e.g., missing exporter endpoint, missing metric names, or unavailable counters).

Example interpretation from a 10s run:
- `Client TPS: 36.80` means the client observed ~36.8 completion tokens/sec.
- `GPU power: 18.5 W` and `Tok/W: 1.990` means ~1.99 tokens per joule during the measurement window.
- If `KV util` is missing, check your vLLM `/metrics` output for metrics containing `kv_cache` / `kv` (and ensure you’re pointing `VLLM_METRICS_URL` at the correct `/metrics` endpoint).

## Latency benchmark (`latency.py`)

Example:

```bash
python latency.py \
  --repeat 10 \
  --warmup 1 \
  --out latency_results.json
```

To disable streaming (no TTFT / ITL):

```bash
python latency.py --no-stream --out latency_results_no_stream.json
```

## Metrics glossary

Below are the definitions for metrics reported by `concurrency.py` and `latency.py`.

### Shared concepts

- **Concurrency level (C)**: number of parallel “sessions” (workers) sending requests.
- **End-to-end latency (E2E)**: time from request start to receiving the final token (or final response).
- **TTFT (Time To First Token)**: time from request start to the first streamed token.
  - TTFT is a useful *proxy* for queueing delay when using streaming.

### `concurrency.py` metrics

Per concurrency level, the script runs a fixed **measurement window** of `--duration-s`. Workers stop launching new requests when the window ends, but already-started requests are allowed to finish.

- **RPS (Requests per second)**
  - Definition: `completed_requests / duration_s`
  - Example: if 900 requests complete over 60s, RPS = `900 / 60 = 15`.

- **TPS (Tokens per second)**
  - Definition: `sum(completion_tokens) / duration_s`
  - Notes:
    - Prefers server-reported `usage.completion_tokens` from the final streaming chunk.
    - Falls back to `tiktoken` token estimation if usage is missing.
  - Example: if 540,000 completion tokens are generated over 60s, TPS = `540000 / 60 = 9000`.

- **Tail latency under load**
  - Reported as percentiles, e.g. `E2E p95`, `TTFT p95`, `p99`.
  - Example: `E2E p95 = 1200 ms` means 95% of completed requests finished in ≤1.2s.

- **Error count / error rate**
  - Definition: number of requests that raised an exception or timed out.

- **Queue wait time (from vLLM `/metrics`, optional)**
  - If available, `queue_time_avg_s` is computed from counters:
    - `delta(queue_time_seconds_sum) / delta(queue_time_seconds_count)`
  - Interpretation: average time requests spent waiting in the server queue.

- **Waiting / running request gauges (from vLLM `/metrics`, optional)**
  - `waiting_mean`, `waiting_max`: typical / peak queued requests during the run window.
  - `running_mean`, `running_max`: typical / peak in-flight running requests.
  - These are helpful to estimate **max concurrent sessions actually achieved**.

- **Server TPS (from vLLM `/metrics`, optional)**
  - If the server exposes token counters, the benchmark reports `server_tps` as:
    - `delta(generation_tokens_total) / duration_s`
  - Use it to cross-check client-side TPS.

- **Saturation point (heuristic)**
  - The script prints a “saturation” concurrency when it detects:
    - a sharp p95 E2E increase (e.g. ≥50% jump), or
    - diminishing RPS gains while p95 latency still rises.
  - Interpretation: the load level where adding more concurrency mostly increases queueing/latency instead of throughput.

### `latency.py` metrics

`latency.py` measures per-request latency for a repeated prompt.

- **TTFT**
  - Only available in streaming mode.

- **End-to-end latency (E2E)**
  - Always available.

- **ITL (Inter-token latency)**
  - Definition: the time gaps between streamed token arrivals.
  - Reported as p50/p95/p99 over all observed token gaps.

- **Max gap / stall time (streaming)**
  - `max_gap_s`: largest inter-token gap within a request.
  - `stall_time_s`: sum of `(gap - stall_threshold)` over gaps larger than `--stall-threshold-ms`.

- **Tokens/sec**
  - Approximate decode rate computed from completion token count and generation time.

## Notes / tips

- If you downloaded a local dataset subset into `data/HuggingFaceH4__ultrachat_200k/...`, you can delete `data/hf_cache/`.
- If you’re using ngrok for the OpenAI `/v1` endpoint, the vLLM `/metrics` endpoint is often available on the same tunnel as `/metrics`.

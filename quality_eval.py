from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
import random
import re
import shutil
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv

from utils import dataset_id


DATASET_SPECS: Dict[str, Dict[str, Any]] = {
    "cnn_dailymail": {
        "hf_id": "cnn_dailymail",
        "config": "3.0.0",
        "default_split": "validation",
        "input_field": "article",
        "target_field": "highlights",
        "instruction": "Summarize the following news article in a concise paragraph.",
    },
    "xsum": {
        "hf_id": "xsum",
        "config": None,
        "default_split": "validation",
        "input_field": "document",
        "target_field": "summary",
        "instruction": "Summarize the following article in one short abstractive summary.",
    },
}

PERPLEXITY_SPECS: Dict[str, Dict[str, Any]] = {
    "wikitext": {
        "hf_id": "wikitext",
        "config": "wikitext-2-raw-v1",
        "default_split": "test",
        "text_field": "text",
    }
}


@dataclass
class QualityExample:
    source: str
    reference: str
    prediction: str


@dataclass
class QualitySummary:
    dataset: str
    split: str
    sample_count: int
    rouge1_f1: Optional[float]
    rouge2_f1: Optional[float]
    rougeL_f1: Optional[float]
    rougeLsum_f1: Optional[float]
    bertscore_f1: Optional[float]
    bleu: Optional[float]
    chrf: Optional[float]
    bleurt: Optional[float]
    bartscore: Optional[float]
    compression_ratio: Optional[float]
    extractive_coverage: Optional[float]
    extractive_density: Optional[float]
    novel_1gram_ratio: Optional[float]
    novel_2gram_ratio: Optional[float]
    repeated_2gram_ratio: Optional[float]
    repeated_3gram_ratio: Optional[float]
    entity_precision: Optional[float]
    entity_recall: Optional[float]
    unsupported_entity_rate: Optional[float]


@dataclass
class PerplexityProbe:
    supported: bool
    endpoint: str
    detail: str


@dataclass
class PerplexitySummary:
    dataset: str
    split: str
    sample_count: int
    token_count: int
    avg_nll: Optional[float]
    perplexity: Optional[float]


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?")
_ENTITY_RE = re.compile(
    r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,4}|[A-Z]{2,}(?:\s+[A-Z]{2,}){0,4})\b"
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _force_hf_cache_under(base_dir: str) -> str:
    cache_root = os.path.join(base_dir, "hf_cache")
    hub_cache = os.path.join(cache_root, "hub")
    datasets_cache = os.path.join(cache_root, "datasets")

    os.makedirs(hub_cache, exist_ok=True)
    os.makedirs(datasets_cache, exist_ok=True)

    os.environ["HF_HOME"] = cache_root
    os.environ["HF_HUB_CACHE"] = hub_cache
    os.environ["HUGGINGFACE_HUB_CACHE"] = hub_cache
    os.environ["HF_DATASETS_CACHE"] = datasets_cache
    return cache_root


def _dataset_disk_name(hf_id: str, config: Optional[str]) -> str:
    if config:
        return dataset_id(f"{hf_id}__{config}")
    return dataset_id(hf_id)


def _maybe_configure_hf_insecure(hf_insecure: bool) -> None:
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


def _configure_hf_runtime(
    *, hf_insecure: bool, hf_token: Optional[str]
) -> None:
    token = (hf_token or "").strip()
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = token
    _maybe_configure_hf_insecure(hf_insecure)


def _build_chat_prompt(instruction: str, source_text: str) -> str:
    return f"{instruction}\n\nArticle:\n{source_text.strip()}\n\nSummary:"


def _cap_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars]


def _mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def _tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]


def _ngrams(tokens: Sequence[str], n: int) -> List[Tuple[str, ...]]:
    if n <= 0 or len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _repeated_ngram_ratio(tokens: Sequence[str], n: int) -> Optional[float]:
    grams = _ngrams(tokens, n)
    if not grams:
        return None
    counts = Counter(grams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return float(repeated / len(grams))


def _extractive_fragment_lengths(
    source_tokens: Sequence[str], summary_tokens: Sequence[str]
) -> List[int]:
    fragments: List[int] = []
    i = 0
    while i < len(summary_tokens):
        best = 0
        for j in range(len(source_tokens)):
            k = 0
            while (
                i + k < len(summary_tokens)
                and j + k < len(source_tokens)
                and summary_tokens[i + k] == source_tokens[j + k]
            ):
                k += 1
            if k > best:
                best = k
        if best > 0:
            fragments.append(best)
            i += best
        else:
            i += 1
    return fragments


def _extractiveness_metrics(
    source_text: str, summary_text: str
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    source_tokens = _tokenize(source_text)
    summary_tokens = _tokenize(summary_text)
    if not source_tokens or not summary_tokens:
        return None, None, None, None

    fragments = _extractive_fragment_lengths(source_tokens, summary_tokens)
    coverage = sum(fragments) / len(summary_tokens)
    density = sum(length * length for length in fragments) / len(
        summary_tokens
    )

    source_unigrams = set(_ngrams(source_tokens, 1))
    source_bigrams = set(_ngrams(source_tokens, 2))
    summary_unigrams = _ngrams(summary_tokens, 1)
    summary_bigrams = _ngrams(summary_tokens, 2)

    novel_1gram_ratio = None
    if summary_unigrams:
        novel_1gram_ratio = float(
            sum(1 for gram in summary_unigrams if gram not in source_unigrams)
            / len(summary_unigrams)
        )

    novel_2gram_ratio = None
    if summary_bigrams:
        novel_2gram_ratio = float(
            sum(1 for gram in summary_bigrams if gram not in source_bigrams)
            / len(summary_bigrams)
        )

    return (
        float(coverage),
        float(density),
        novel_1gram_ratio,
        novel_2gram_ratio,
    )


def _compression_ratio(source_text: str, summary_text: str) -> Optional[float]:
    source_tokens = _tokenize(source_text)
    summary_tokens = _tokenize(summary_text)
    if not source_tokens or not summary_tokens:
        return None
    return float(len(summary_tokens) / len(source_tokens))


def _extract_entities(text: str) -> List[str]:
    entities = {
        " ".join(match.group(0).split()).casefold()
        for match in _ENTITY_RE.finditer(text)
        if len(match.group(0).strip()) > 1
    }
    return sorted(entities)


def _summary_for_rouge_lsum(text: str) -> str:
    text = " ".join(text.split())
    if not text:
        return text
    parts = [
        part.strip() for part in _SENTENCE_SPLIT_RE.split(text) if part.strip()
    ]
    if not parts:
        return text
    return "\n".join(parts)


def _entity_faithfulness(
    source_text: str, summary_text: str
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    source_entities = set(_extract_entities(source_text))
    summary_entities = set(_extract_entities(summary_text))
    if not source_entities and not summary_entities:
        return None, None, None

    matched = source_entities & summary_entities
    precision = (
        float(len(matched) / len(summary_entities))
        if summary_entities
        else None
    )
    recall = (
        float(len(matched) / len(source_entities)) if source_entities else None
    )
    unsupported_rate = (
        float((len(summary_entities) - len(matched)) / len(summary_entities))
        if summary_entities
        else None
    )
    return precision, recall, unsupported_rate


def _compute_bleurt(
    *, predictions: Sequence[str], references: Sequence[str], checkpoint: str
) -> Optional[float]:
    try:
        from bleurt import score as bleurt_score  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Missing dependency 'BLEURT'. Install it with: python -m pip install git+https://github.com/google-research/bleurt.git"
        ) from e

    resolved_checkpoint = checkpoint.strip()
    scorer = None
    try:
        if not resolved_checkpoint or resolved_checkpoint.lower() == "default":
            scorer = bleurt_score.BleurtScorer()
        else:
            if not os.path.isdir(resolved_checkpoint):
                raise RuntimeError(
                    "BLEURT expects a local checkpoint directory. Leave --bleurt-checkpoint empty or 'default' to use the bundled checkpoint, or pass a local exported BLEURT checkpoint path."
                )
            scorer = bleurt_score.BleurtScorer(checkpoint=resolved_checkpoint)
        scores = scorer.score(
            references=list(references), candidates=list(predictions)
        )
    finally:
        predictor = getattr(scorer, "_predictor", None)
        if predictor is not None:
            close_fn = getattr(predictor, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass

    if not isinstance(scores, list) or not scores:
        raise RuntimeError(f"Unexpected BLEURT score payload: {scores}")
    return _mean([float(score) for score in scores])


def _compute_bartscore(
    *,
    sources: Sequence[str],
    predictions: Sequence[str],
    model_name: str,
    device: str,
) -> Optional[float]:
    try:
        import torch  # type: ignore
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "BARTScore requires torch and transformers. Install them with: pip install torch transformers"
        ) from e

    resolved_device = device
    if device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(resolved_device)
    model.eval()

    scores: List[float] = []
    with torch.no_grad():
        for source, prediction in zip(sources, predictions):
            source_inputs = tokenizer(
                source,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
            )
            target_inputs = tokenizer(
                text_target=prediction,
                return_tensors="pt",
                truncation=True,
                max_length=256,
            )
            labels = target_inputs["input_ids"].clone()
            labels[labels == tokenizer.pad_token_id] = -100
            outputs = model(
                input_ids=source_inputs["input_ids"].to(resolved_device),
                attention_mask=source_inputs["attention_mask"].to(
                    resolved_device
                ),
                labels=labels.to(resolved_device),
            )
            scores.append(float(-outputs.loss.item()))
    return _mean(scores)


def _load_local_or_stream_subset(
    *,
    hf_id: str,
    config: Optional[str],
    split: str,
    data_dir: str,
    subset_rows: int,
    seed: int,
    hf_insecure: bool,
) -> Any:
    try:
        from datasets import Dataset, load_dataset, load_from_disk  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Missing dependency 'datasets'. Install it with: pip install datasets"
        ) from e

    cache_dir = _force_hf_cache_under(data_dir)
    local_path = os.path.join(
        os.path.abspath(data_dir), _dataset_disk_name(hf_id, config), split
    )

    if os.path.isdir(local_path):
        ds = load_from_disk(local_path)
        if len(ds) >= subset_rows:
            return ds
        print(
            f"Local subset at {local_path} only has {len(ds)} rows; refreshing to {subset_rows} rows."
        )
        shutil.rmtree(local_path, ignore_errors=True)

    if subset_rows <= 0:
        raise RuntimeError(
            "A positive subset size is required when no local dataset copy exists."
        )

    _maybe_configure_hf_insecure(hf_insecure)
    ds_iter = load_dataset(
        hf_id,
        config,
        split=split,
        streaming=True,
        cache_dir=cache_dir,
    )

    try:
        ds_iter = ds_iter.shuffle(
            seed=int(seed), buffer_size=min(10_000, max(200, subset_rows * 20))
        )
    except Exception:
        pass

    rows: List[Dict[str, Any]] = []
    for row in ds_iter:
        if isinstance(row, dict):
            rows.append(row)
        if len(rows) >= subset_rows:
            break

    if not rows:
        raise RuntimeError(
            f"No rows could be materialized for {hf_id} [{split}] via streaming."
        )

    ds = Dataset.from_list(rows)
    os.makedirs(local_path, exist_ok=True)
    ds.save_to_disk(local_path)
    return ds


def _prepare_quality_examples(
    *,
    dataset_name: str,
    split: str,
    data_dir: str,
    sample_count: int,
    subset_rows: int,
    seed: int,
    source_max_chars: int,
    hf_insecure: bool,
) -> List[Tuple[str, str]]:
    spec = DATASET_SPECS.get(dataset_name)
    if spec is None:
        raise RuntimeError(
            f"Unsupported dataset '{dataset_name}'. Choose from: {', '.join(sorted(DATASET_SPECS))}."
        )

    ds = _load_local_or_stream_subset(
        hf_id=str(spec["hf_id"]),
        config=spec.get("config"),
        split=split,
        data_dir=data_dir,
        subset_rows=max(subset_rows, sample_count),
        seed=seed,
        hf_insecure=hf_insecure,
    )

    pairs: List[Tuple[str, str]] = []
    indices = list(range(len(ds)))
    random.Random(seed).shuffle(indices)
    for idx in indices:
        row = ds[int(idx)]
        source = row.get(spec["input_field"])
        target = row.get(spec["target_field"])
        if not isinstance(source, str) or not source.strip():
            continue
        if not isinstance(target, str) or not target.strip():
            continue
        pairs.append(
            (_cap_text(source.strip(), source_max_chars), target.strip())
        )
        if len(pairs) >= sample_count:
            break

    if len(pairs) < sample_count:
        raise RuntimeError(
            f"Only prepared {len(pairs)} examples from {dataset_name} [{split}]; requested {sample_count}."
        )
    return pairs


def _prepare_perplexity_texts(
    *,
    dataset_name: str,
    split: str,
    data_dir: str,
    sample_count: int,
    subset_rows: int,
    seed: int,
    text_max_chars: int,
    hf_insecure: bool,
) -> List[str]:
    spec = PERPLEXITY_SPECS.get(dataset_name)
    if spec is None:
        raise RuntimeError(
            f"Unsupported perplexity dataset '{dataset_name}'. Choose from: {', '.join(sorted(PERPLEXITY_SPECS))}."
        )

    ds = _load_local_or_stream_subset(
        hf_id=str(spec["hf_id"]),
        config=spec.get("config"),
        split=split,
        data_dir=data_dir,
        subset_rows=max(subset_rows, sample_count),
        seed=seed,
        hf_insecure=hf_insecure,
    )

    texts: List[str] = []
    indices = list(range(len(ds)))
    random.Random(seed).shuffle(indices)
    for idx in indices:
        row = ds[int(idx)]
        text = row.get(spec["text_field"])
        if not isinstance(text, str):
            continue
        text = _cap_text(text.strip(), text_max_chars)
        if not text:
            continue
        texts.append(text)
        if len(texts) >= sample_count:
            break

    if len(texts) < sample_count:
        raise RuntimeError(
            f"Only prepared {len(texts)} perplexity texts from {dataset_name} [{split}]; requested {sample_count}."
        )
    return texts


def _make_openai_client(*, base_url: str, api_key: str, insecure: bool) -> Any:
    from openai import OpenAI

    if insecure:
        try:
            import httpx  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "--insecure requested, but httpx is not available. Install with: pip install httpx"
            ) from e
        return OpenAI(
            base_url=base_url,
            api_key=api_key,
            http_client=httpx.Client(verify=False),
        )

    return OpenAI(base_url=base_url, api_key=api_key)


def _generate_summaries(
    *,
    client: Any,
    model: str,
    dataset_name: str,
    pairs: Sequence[Tuple[str, str]],
    temperature: float,
    max_tokens: int,
) -> List[QualityExample]:
    spec = DATASET_SPECS[dataset_name]
    out: List[QualityExample] = []

    for source, reference in pairs:
        prompt = _build_chat_prompt(str(spec["instruction"]), source)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        prediction = response.choices[0].message.content or ""
        out.append(
            QualityExample(
                source=source,
                reference=reference,
                prediction=prediction.strip(),
            )
        )

    return out


def _score_quality(
    *,
    examples: Sequence[QualityExample],
    include_bleu: bool,
    include_bertscore: bool,
    include_bleurt: bool,
    bleurt_checkpoint: str,
    include_bartscore: bool,
    bartscore_model: str,
    bartscore_device: str,
) -> QualitySummary:
    try:
        from rouge_score import rouge_scorer  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Missing dependency 'rouge-score'. Install it with: pip install rouge-score"
        ) from e

    rouge = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL", "rougeLsum"],
        use_stemmer=True,
        split_summaries=False,
    )
    rouge1_vals: List[float] = []
    rouge2_vals: List[float] = []
    rougeL_vals: List[float] = []
    rougeLsum_vals: List[float] = []
    compression_vals: List[float] = []
    coverage_vals: List[float] = []
    density_vals: List[float] = []
    novel_1gram_vals: List[float] = []
    novel_2gram_vals: List[float] = []
    repeated_2gram_vals: List[float] = []
    repeated_3gram_vals: List[float] = []
    entity_precision_vals: List[float] = []
    entity_recall_vals: List[float] = []
    unsupported_entity_vals: List[float] = []
    refs = [ex.reference for ex in examples]
    preds = [ex.prediction for ex in examples]
    sources = [ex.source for ex in examples]

    for ex in examples:
        scores = rouge.score(
            _summary_for_rouge_lsum(ex.reference),
            _summary_for_rouge_lsum(ex.prediction),
        )
        rouge1_vals.append(float(scores["rouge1"].fmeasure))
        rouge2_vals.append(float(scores["rouge2"].fmeasure))
        rougeL_vals.append(float(scores["rougeL"].fmeasure))
        rougeLsum_vals.append(float(scores["rougeLsum"].fmeasure))

        compression = _compression_ratio(ex.source, ex.prediction)
        if compression is not None:
            compression_vals.append(compression)

        coverage, density, novel_1, novel_2 = _extractiveness_metrics(
            ex.source, ex.prediction
        )
        if coverage is not None:
            coverage_vals.append(coverage)
        if density is not None:
            density_vals.append(density)
        if novel_1 is not None:
            novel_1gram_vals.append(novel_1)
        if novel_2 is not None:
            novel_2gram_vals.append(novel_2)

        prediction_tokens = _tokenize(ex.prediction)
        repeated_2gram = _repeated_ngram_ratio(prediction_tokens, 2)
        repeated_3gram = _repeated_ngram_ratio(prediction_tokens, 3)
        if repeated_2gram is not None:
            repeated_2gram_vals.append(repeated_2gram)
        if repeated_3gram is not None:
            repeated_3gram_vals.append(repeated_3gram)

        entity_precision, entity_recall, unsupported_entity_rate = (
            _entity_faithfulness(ex.source, ex.prediction)
        )
        if entity_precision is not None:
            entity_precision_vals.append(entity_precision)
        if entity_recall is not None:
            entity_recall_vals.append(entity_recall)
        if unsupported_entity_rate is not None:
            unsupported_entity_vals.append(unsupported_entity_rate)

    bertscore_f1: Optional[float] = None
    if include_bertscore:
        try:
            from bert_score import score as bert_score  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Missing dependency 'bert-score'. Install it with: pip install bert-score"
            ) from e
        _, _, f1 = bert_score(preds, refs, lang="en", verbose=False)
        bertscore_f1 = float(f1.mean().item())

    bleu_score: Optional[float] = None
    chrf_score: Optional[float] = None
    if include_bleu:
        try:
            import sacrebleu  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Missing dependency 'sacrebleu'. Install it with: pip install sacrebleu"
            ) from e
        bleu_score = float(sacrebleu.corpus_bleu(preds, [refs]).score)
        chrf_score = float(sacrebleu.corpus_chrf(preds, [refs]).score)
    else:
        try:
            import sacrebleu  # type: ignore

            chrf_score = float(sacrebleu.corpus_chrf(preds, [refs]).score)
        except Exception:
            chrf_score = None

    bleurt_score: Optional[float] = None
    if include_bleurt:
        bleurt_score = _compute_bleurt(
            predictions=preds,
            references=refs,
            checkpoint=bleurt_checkpoint,
        )

    bartscore: Optional[float] = None
    if include_bartscore:
        bartscore = _compute_bartscore(
            sources=sources,
            predictions=preds,
            model_name=bartscore_model,
            device=bartscore_device,
        )

    return QualitySummary(
        dataset="",
        split="",
        sample_count=len(examples),
        rouge1_f1=_mean(rouge1_vals),
        rouge2_f1=_mean(rouge2_vals),
        rougeL_f1=_mean(rougeL_vals),
        rougeLsum_f1=_mean(rougeLsum_vals),
        bertscore_f1=bertscore_f1,
        bleu=bleu_score,
        chrf=chrf_score,
        bleurt=bleurt_score,
        bartscore=bartscore,
        compression_ratio=_mean(compression_vals),
        extractive_coverage=_mean(coverage_vals),
        extractive_density=_mean(density_vals),
        novel_1gram_ratio=_mean(novel_1gram_vals),
        novel_2gram_ratio=_mean(novel_2gram_vals),
        repeated_2gram_ratio=_mean(repeated_2gram_vals),
        repeated_3gram_ratio=_mean(repeated_3gram_vals),
        entity_precision=_mean(entity_precision_vals),
        entity_recall=_mean(entity_recall_vals),
        unsupported_entity_rate=_mean(unsupported_entity_vals),
    )


def _completions_url(base_url: str) -> str:
    return base_url.rstrip("/") + "/completions"


def _extract_token_logprobs(choice: Dict[str, Any]) -> List[float]:
    token_logprobs: List[float] = []
    logprobs = choice.get("logprobs")
    if isinstance(logprobs, dict):
        raw = logprobs.get("token_logprobs")
        if isinstance(raw, list):
            for value in raw:
                if isinstance(value, (int, float)):
                    token_logprobs.append(float(value))
        prompt_logprobs = logprobs.get("prompt_logprobs")
        if isinstance(prompt_logprobs, list):
            for item in prompt_logprobs:
                if isinstance(item, dict):
                    score = item.get("logprob")
                    if isinstance(score, (int, float)):
                        token_logprobs.append(float(score))
    prompt_logprobs = choice.get("prompt_logprobs")
    if isinstance(prompt_logprobs, list):
        for item in prompt_logprobs:
            if isinstance(item, dict):
                score = item.get("logprob")
                if isinstance(score, (int, float)):
                    token_logprobs.append(float(score))
    return token_logprobs


def _post_json(
    *,
    url: str,
    api_key: str,
    payload: Dict[str, Any],
    insecure: bool,
    timeout_s: float,
) -> Tuple[int, Any]:
    try:
        import httpx  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Missing dependency 'httpx'. Install it with: pip install httpx"
        ) from e

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    with httpx.Client(verify=not insecure, timeout=timeout_s) as client:
        response = client.post(url, headers=headers, json=payload)
        try:
            body = response.json()
        except Exception:
            body = response.text
        return int(response.status_code), body


def probe_api_perplexity_support(
    *,
    base_url: str,
    api_key: str,
    model: str,
    insecure: bool,
    timeout_s: float,
) -> PerplexityProbe:
    endpoint = _completions_url(base_url)
    payload = {
        "model": model,
        "prompt": "Perplexity probe.",
        "max_tokens": 0,
        "temperature": 0,
        "echo": True,
        "logprobs": 1,
        "prompt_logprobs": 1,
    }
    status, body = _post_json(
        url=endpoint,
        api_key=api_key,
        payload=payload,
        insecure=insecure,
        timeout_s=timeout_s,
    )
    if status >= 400:
        return PerplexityProbe(
            supported=False,
            endpoint=endpoint,
            detail=f"HTTP {status}: {body}",
        )
    if not isinstance(body, dict):
        return PerplexityProbe(
            supported=False,
            endpoint=endpoint,
            detail=f"Unexpected response body: {body}",
        )
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return PerplexityProbe(
            supported=False,
            endpoint=endpoint,
            detail="No choices returned from /completions.",
        )
    token_logprobs = _extract_token_logprobs(choices[0])
    if not token_logprobs:
        return PerplexityProbe(
            supported=False,
            endpoint=endpoint,
            detail="The endpoint did not return prompt token logprobs needed for perplexity.",
        )
    return PerplexityProbe(
        supported=True,
        endpoint=endpoint,
        detail=f"Prompt logprobs available for {len(token_logprobs)} tokens.",
    )


def _measure_api_perplexity(
    *,
    base_url: str,
    api_key: str,
    model: str,
    texts: Sequence[str],
    insecure: bool,
    timeout_s: float,
) -> PerplexitySummary:
    endpoint = _completions_url(base_url)
    total_neg_logprob = 0.0
    total_tokens = 0

    for text in texts:
        payload = {
            "model": model,
            "prompt": text,
            "max_tokens": 0,
            "temperature": 0,
            "echo": True,
            "logprobs": 1,
            "prompt_logprobs": 1,
        }
        status, body = _post_json(
            url=endpoint,
            api_key=api_key,
            payload=payload,
            insecure=insecure,
            timeout_s=timeout_s,
        )
        if status >= 400:
            raise RuntimeError(
                f"Perplexity request failed with HTTP {status}: {body}"
            )
        if not isinstance(body, dict):
            raise RuntimeError(f"Unexpected perplexity response body: {body}")
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("Perplexity response did not contain a choice.")

        token_logprobs = _extract_token_logprobs(choices[0])
        if not token_logprobs:
            raise RuntimeError(
                "Perplexity response did not contain usable prompt token logprobs."
            )
        total_neg_logprob += -sum(token_logprobs)
        total_tokens += len(token_logprobs)

    avg_nll = (total_neg_logprob / total_tokens) if total_tokens > 0 else None
    ppl = math.exp(avg_nll) if avg_nll is not None else None
    return PerplexitySummary(
        dataset="",
        split="",
        sample_count=len(texts),
        token_count=total_tokens,
        avg_nll=avg_nll,
        perplexity=ppl,
    )


def _print_quality(summary: QualitySummary) -> None:
    print("Quality metrics")
    print(f"  dataset:      {summary.dataset} [{summary.split}]")
    print(f"  samples:      {summary.sample_count}")
    print(
        f"  rouge1_f1:    {summary.rouge1_f1:.4f}"
        if summary.rouge1_f1 is not None
        else "  rouge1_f1:    n/a"
    )
    print(
        f"  rouge2_f1:    {summary.rouge2_f1:.4f}"
        if summary.rouge2_f1 is not None
        else "  rouge2_f1:    n/a"
    )
    print(
        f"  rougeL_f1:    {summary.rougeL_f1:.4f}"
        if summary.rougeL_f1 is not None
        else "  rougeL_f1:    n/a"
    )
    print(
        f"  rougeLsum_f1: {summary.rougeLsum_f1:.4f}"
        if summary.rougeLsum_f1 is not None
        else "  rougeLsum_f1: n/a"
    )
    print(
        f"  bertscore_f1: {summary.bertscore_f1:.4f}"
        if summary.bertscore_f1 is not None
        else "  bertscore_f1: n/a"
    )
    print(
        f"  bleu:         {summary.bleu:.2f}"
        if summary.bleu is not None
        else "  bleu:         n/a"
    )
    print(
        f"  chrf:         {summary.chrf:.2f}"
        if summary.chrf is not None
        else "  chrf:         n/a"
    )
    print(
        f"  bleurt:       {summary.bleurt:.4f}"
        if summary.bleurt is not None
        else "  bleurt:       n/a"
    )
    print(
        f"  bartscore:    {summary.bartscore:.4f}"
        if summary.bartscore is not None
        else "  bartscore:    n/a"
    )
    print(
        f"  compress_rt:  {summary.compression_ratio:.4f}"
        if summary.compression_ratio is not None
        else "  compress_rt:  n/a"
    )
    print(
        f"  ext_coverage: {summary.extractive_coverage:.4f}"
        if summary.extractive_coverage is not None
        else "  ext_coverage: n/a"
    )
    print(
        f"  ext_density:  {summary.extractive_density:.4f}"
        if summary.extractive_density is not None
        else "  ext_density:  n/a"
    )
    print(
        f"  novel_1gram:  {summary.novel_1gram_ratio:.4f}"
        if summary.novel_1gram_ratio is not None
        else "  novel_1gram:  n/a"
    )
    print(
        f"  novel_2gram:  {summary.novel_2gram_ratio:.4f}"
        if summary.novel_2gram_ratio is not None
        else "  novel_2gram:  n/a"
    )
    print(
        f"  repeat_2gram: {summary.repeated_2gram_ratio:.4f}"
        if summary.repeated_2gram_ratio is not None
        else "  repeat_2gram: n/a"
    )
    print(
        f"  repeat_3gram: {summary.repeated_3gram_ratio:.4f}"
        if summary.repeated_3gram_ratio is not None
        else "  repeat_3gram: n/a"
    )
    print(
        f"  ent_prec:     {summary.entity_precision:.4f}"
        if summary.entity_precision is not None
        else "  ent_prec:     n/a"
    )
    print(
        f"  ent_recall:   {summary.entity_recall:.4f}"
        if summary.entity_recall is not None
        else "  ent_recall:   n/a"
    )
    print(
        f"  ent_unsupported:{summary.unsupported_entity_rate:.4f}"
        if summary.unsupported_entity_rate is not None
        else "  ent_unsupported:n/a"
    )


def _print_perplexity_probe(probe: PerplexityProbe) -> None:
    print("Perplexity capability")
    print(f"  endpoint:     {probe.endpoint}")
    print(f"  supported:    {probe.supported}")
    print(f"  detail:       {probe.detail}")


def _print_perplexity(summary: PerplexitySummary) -> None:
    print("Perplexity")
    print(f"  dataset:      {summary.dataset} [{summary.split}]")
    print(f"  samples:      {summary.sample_count}")
    print(f"  tokens:       {summary.token_count}")
    print(
        f"  avg_nll:      {summary.avg_nll:.6f}"
        if summary.avg_nll is not None
        else "  avg_nll:      n/a"
    )
    print(
        f"  perplexity:   {summary.perplexity:.4f}"
        if summary.perplexity is not None
        else "  perplexity:   n/a"
    )


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Evaluate summarization quality and optional API-side perplexity for an OpenAI-compatible vLLM endpoint"
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
        "--dataset",
        choices=sorted(DATASET_SPECS),
        default=os.getenv("QUALITY_DATASET", "cnn_dailymail"),
    )
    parser.add_argument(
        "--split",
        default=os.getenv("QUALITY_SPLIT", "validation"),
        help="Summarization dataset split. cnn_dailymail and xsum commonly use validation/test.",
    )
    parser.add_argument(
        "--data-dir",
        default=os.getenv("DATA_DIR", "data"),
        help="Base folder for local datasets and HF cache (default: ./data)",
    )
    parser.add_argument(
        "--quality-samples",
        type=int,
        default=int(os.getenv("QUALITY_SAMPLES", "25")),
        help="Number of summarization examples to score",
    )
    parser.add_argument(
        "--subset-rows",
        type=int,
        default=int(os.getenv("QUALITY_SUBSET_ROWS", "50")),
        help="Rows to materialize locally under ./data when no saved subset exists",
    )
    parser.add_argument(
        "--source-max-chars",
        type=int,
        default=int(os.getenv("SOURCE_MAX_CHARS", "6000")),
        help="Cap source article length before sending to the model",
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
        "--sample-seed",
        type=int,
        default=int(os.getenv("SAMPLE_SEED", "123")),
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS verification for the vLLM endpoint and HF streaming downloads",
    )
    parser.add_argument(
        "--hf-token",
        default=os.getenv("HF_TOKEN", os.getenv("HUGGING_FACE_HUB_TOKEN", "")),
        help="Optional Hugging Face token for model and metric downloads.",
    )
    parser.add_argument(
        "--skip-bertscore",
        action="store_true",
        help="Skip BERTScore if you only want lexical metrics",
    )
    parser.add_argument(
        "--include-bleu",
        action="store_true",
        help="Also compute BLEU. This is optional for summarization.",
    )
    parser.add_argument(
        "--include-bleurt",
        action="store_true",
        help="Compute BLEURT using the Hugging Face evaluate metric. This is optional and heavier.",
    )
    parser.add_argument(
        "--bleurt-checkpoint",
        default=os.getenv("BLEURT_CHECKPOINT", "default"),
        help="BLEURT checkpoint directory, or 'default' to use the bundled BLEURT checkpoint.",
    )
    parser.add_argument(
        "--include-bartscore",
        action="store_true",
        help="Compute a source-to-summary BARTScore-style metric. This is optional and heavier.",
    )
    parser.add_argument(
        "--bartscore-model",
        default=os.getenv("BARTSCORE_MODEL", "facebook/bart-large-cnn"),
        help="Seq2seq model used for BARTScore-style scoring.",
    )
    parser.add_argument(
        "--bartscore-device",
        default=os.getenv("BARTSCORE_DEVICE", "auto"),
        help="Device for BARTScore-style scoring: auto, cpu, or cuda.",
    )
    parser.add_argument(
        "--skip-perplexity",
        action="store_true",
        help="Skip the API-side perplexity probe and measurement",
    )
    parser.add_argument(
        "--perplexity-dataset",
        choices=sorted(PERPLEXITY_SPECS),
        default=os.getenv("PERPLEXITY_DATASET", "wikitext"),
    )
    parser.add_argument(
        "--perplexity-split",
        default=os.getenv("PERPLEXITY_SPLIT", "test"),
    )
    parser.add_argument(
        "--perplexity-samples",
        type=int,
        default=int(os.getenv("PERPLEXITY_SAMPLES", "20")),
    )
    parser.add_argument(
        "--perplexity-subset-rows",
        type=int,
        default=int(os.getenv("PERPLEXITY_SUBSET_ROWS", "40")),
    )
    parser.add_argument(
        "--perplexity-text-max-chars",
        type=int,
        default=int(os.getenv("PERPLEXITY_TEXT_MAX_CHARS", "1200")),
    )
    parser.add_argument(
        "--http-timeout-s",
        type=float,
        default=float(os.getenv("HTTP_TIMEOUT_S", "120")),
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional JSON output path",
    )

    args = parser.parse_args()

    data_dir = os.path.abspath(str(args.data_dir))
    os.makedirs(data_dir, exist_ok=True)
    _force_hf_cache_under(data_dir)
    _configure_hf_runtime(
        hf_insecure=bool(args.insecure), hf_token=str(args.hf_token)
    )

    print(f"Target:  {args.base_url}")
    print(f"Model:   {args.model}")
    print(f"Data:    {data_dir}")
    print(
        f"Quality: {args.dataset} [{args.split}] samples={args.quality_samples}"
    )
    if not args.skip_perplexity:
        print(
            f"PPL:     {args.perplexity_dataset} [{args.perplexity_split}] samples={args.perplexity_samples}"
        )
    print("")

    try:
        pairs = _prepare_quality_examples(
            dataset_name=str(args.dataset),
            split=str(args.split),
            data_dir=data_dir,
            sample_count=int(args.quality_samples),
            subset_rows=int(args.subset_rows),
            seed=int(args.sample_seed),
            source_max_chars=int(args.source_max_chars),
            hf_insecure=bool(args.insecure),
        )
        client = _make_openai_client(
            base_url=str(args.base_url),
            api_key=str(args.api_key),
            insecure=bool(args.insecure),
        )
        examples = _generate_summaries(
            client=client,
            model=str(args.model),
            dataset_name=str(args.dataset),
            pairs=pairs,
            temperature=float(args.temperature),
            max_tokens=int(args.max_tokens),
        )
        quality = _score_quality(
            examples=examples,
            include_bleu=bool(args.include_bleu),
            include_bertscore=not bool(args.skip_bertscore),
            include_bleurt=bool(args.include_bleurt),
            bleurt_checkpoint=str(args.bleurt_checkpoint),
            include_bartscore=bool(args.include_bartscore),
            bartscore_model=str(args.bartscore_model),
            bartscore_device=str(args.bartscore_device),
        )
        quality.dataset = str(args.dataset)
        quality.split = str(args.split)
        _print_quality(quality)
    except KeyboardInterrupt:
        print("Interrupted.")
        return 130
    except Exception as e:
        print(f"Quality evaluation failed: {e}", file=sys.stderr)
        return 2

    probe: Optional[PerplexityProbe] = None
    ppl: Optional[PerplexitySummary] = None
    if not args.skip_perplexity:
        try:
            probe = probe_api_perplexity_support(
                base_url=str(args.base_url),
                api_key=str(args.api_key),
                model=str(args.model),
                insecure=bool(args.insecure),
                timeout_s=float(args.http_timeout_s),
            )
            print("")
            _print_perplexity_probe(probe)
            if probe.supported:
                texts = _prepare_perplexity_texts(
                    dataset_name=str(args.perplexity_dataset),
                    split=str(args.perplexity_split),
                    data_dir=data_dir,
                    sample_count=int(args.perplexity_samples),
                    subset_rows=int(args.perplexity_subset_rows),
                    seed=int(args.sample_seed),
                    text_max_chars=int(args.perplexity_text_max_chars),
                    hf_insecure=bool(args.insecure),
                )
                ppl = _measure_api_perplexity(
                    base_url=str(args.base_url),
                    api_key=str(args.api_key),
                    model=str(args.model),
                    texts=texts,
                    insecure=bool(args.insecure),
                    timeout_s=float(args.http_timeout_s),
                )
                ppl.dataset = str(args.perplexity_dataset)
                ppl.split = str(args.perplexity_split)
                print("")
                _print_perplexity(ppl)
        except KeyboardInterrupt:
            print("Interrupted.")
            return 130
        except Exception as e:
            print(f"Perplexity measurement failed: {e}", file=sys.stderr)
            return 2

    if args.out:
        payload = {
            "config": {
                "base_url": args.base_url,
                "model": args.model,
                "dataset": args.dataset,
                "split": args.split,
                "include_bertscore": not bool(args.skip_bertscore),
                "include_bleu": bool(args.include_bleu),
                "include_bleurt": bool(args.include_bleurt),
                "bleurt_checkpoint": args.bleurt_checkpoint,
                "include_bartscore": bool(args.include_bartscore),
                "bartscore_model": args.bartscore_model,
                "quality_samples": args.quality_samples,
                "subset_rows": args.subset_rows,
                "perplexity_dataset": args.perplexity_dataset,
                "perplexity_split": args.perplexity_split,
                "perplexity_samples": args.perplexity_samples,
                "perplexity_subset_rows": args.perplexity_subset_rows,
                "data_dir": data_dir,
                "insecure": bool(args.insecure),
            },
            "quality": asdict(quality),
            "perplexity_probe": asdict(probe) if probe is not None else None,
            "perplexity": asdict(ppl) if ppl is not None else None,
        }
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print("")
        print(f"Wrote JSON: {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

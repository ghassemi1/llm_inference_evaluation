from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv

from utils import dataset_id


def _force_hf_cache_under(base_dir: str) -> str:
    cache_root = os.path.join(base_dir, "hf_cache")
    hub_cache = os.path.join(cache_root, "hub")
    datasets_cache = os.path.join(cache_root, "datasets")

    os.makedirs(hub_cache, exist_ok=True)
    os.makedirs(datasets_cache, exist_ok=True)

    # Force hub and datasets caches under ./data so HF does not spill into the
    # user's global cache directories.
    os.environ["HF_HOME"] = cache_root
    os.environ["HF_HUB_CACHE"] = hub_cache
    os.environ["HUGGINGFACE_HUB_CACHE"] = hub_cache
    os.environ["HF_DATASETS_CACHE"] = datasets_cache

    return cache_root


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description=(
            "Download a HuggingFace dataset into ./data and save it for local/offline reads. "
            "This uses `datasets.save_to_disk()` so benchmark scripts can load from the data folder."
        )
    )
    parser.add_argument(
        "--dataset",
        default=os.getenv("HF_DATASET", "HuggingFaceH4/ultrachat_200k"),
        help="HuggingFace dataset ID",
    )
    parser.add_argument(
        "--split",
        default=os.getenv("HF_SPLIT", "train_sft"),
        help=(
            "Dataset split to download. For UltraChat, common splits are: "
            "train_sft, test_sft, train_gen, test_gen. "
            "Aliases: 'train' -> 'train_sft', 'test' -> 'test_sft'."
        ),
    )
    parser.add_argument(
        "--data-dir",
        default=os.getenv("DATA_DIR", "data"),
        help="Base folder for downloads (default: ./data)",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help=(
            "Disable TLS certificate verification for HuggingFace downloads. "
            "Use only if you are behind a proxy/SSL-intercept and see CERTIFICATE_VERIFY_FAILED."
        ),
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help=(
            "Optional: limit rows saved to disk (0 = all). Useful if full dataset is too large."
        ),
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help=(
            "When used with --max-rows, stream from HF and materialize only that subset before saving. "
            "This avoids downloading/building the full dataset cache in most cases."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Shuffle seed used only when --max-rows > 0",
    )

    args = parser.parse_args()

    split = str(args.split).strip()
    if split == "train":
        split = "train_sft"
    elif split == "test":
        split = "test_sft"

    try:
        from datasets import Dataset, load_dataset
    except Exception as e:
        print(
            "Missing dependency 'datasets'. Install it with: pip install datasets",
            file=sys.stderr,
        )
        return 2

    data_dir = os.path.abspath(args.data_dir)
    os.makedirs(data_dir, exist_ok=True)

    cache_dir = _force_hf_cache_under(data_dir)

    print(f"Downloading {args.dataset} [{args.split}] …")
    print(f"Cache dir: {cache_dir}")

    if args.insecure:
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
            print(
                "HuggingFace download: TLS verification DISABLED (--insecure)"
            )
        except Exception as e:
            print(
                f"Failed to enable insecure HF download mode: {e}",
                file=sys.stderr,
            )
            return 2

    print(f"Resolved split: {split}")

    if args.streaming:
        if not (args.max_rows and args.max_rows > 0):
            print(
                "--streaming requires --max-rows > 0 (saving an unbounded streaming dataset is not supported).",
                file=sys.stderr,
            )
            return 2

        k = int(args.max_rows)
        print(f"Streaming and materializing {k} rows …")
        ids = load_dataset(args.dataset, split=split, streaming=True)
        try:
            ids = ids.shuffle(
                seed=int(args.seed), buffer_size=min(10_000, k * 20)
            )
        except Exception:
            pass

        rows = []
        for row in ids:
            rows.append(row)
            if len(rows) >= k:
                break

        if len(rows) < k:
            print(
                f"Only received {len(rows)} rows while streaming; requested {k}.",
                file=sys.stderr,
            )
            return 2

        ds = Dataset.from_list(rows)
        print(f"Materialized rows: {len(ds)}")
    else:
        ds = load_dataset(args.dataset, split=split, cache_dir=cache_dir)

        if args.max_rows and args.max_rows > 0:
            n = len(ds)
            k = min(int(args.max_rows), n)
            ds = ds.shuffle(seed=int(args.seed)).select(range(k))
            print(f"Limiting saved rows: {k}/{n}")

    out_path = os.path.join(data_dir, dataset_id(args.dataset), split)
    os.makedirs(out_path, exist_ok=True)

    print(f"Saving to: {out_path}")
    ds.save_to_disk(out_path)

    print("Done.")
    print("")
    print("Next:")
    print(
        "  python concurrency.py --data-dir data --dataset HuggingFaceH4/ultrachat_200k --split train"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

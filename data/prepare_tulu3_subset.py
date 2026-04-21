import argparse
import json
import random
import time
from pathlib import Path

import requests
from tqdm import tqdm


ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"
SIZE_ENDPOINT = "https://datasets-server.huggingface.co/size"
PAGE_SIZE = 100

STANDALONE_SOURCES = {
    "algebra": "allenai/tulu-3-sft-personas-algebra",
    "instruction_following": "allenai/tulu-3-sft-personas-instruction-following",
}

MIXTURE_REPO = "allenai/tulu-3-sft-mixture"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/tulu3_subsets/tulu3_reasoning_small.jsonl")
    parser.add_argument("--algebra", type=int, default=1000)
    parser.add_argument("--instruction-following", type=int, default=1000)
    parser.add_argument("--gsm", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-mixture-scan", type=int, default=250000)
    parser.add_argument("--sleep", type=float, default=0.5)
    return parser.parse_args()


def get_json(url: str, params: dict, retries: int = 5):
    last_error = None
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=60)
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    wait_seconds = int(retry_after)
                else:
                    wait_seconds = min(120, 10 * (attempt + 1))
                print(f"Rate limited by Hugging Face datasets-server. Sleeping {wait_seconds}s before retry.")
                time.sleep(wait_seconds)
                continue
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            last_error = exc
            time.sleep(min(120, 5 * (2 ** attempt)))
    raise RuntimeError(f"Failed request to {url} with params {params}") from last_error


def get_num_rows(dataset: str, config: str = "default", split: str = "train") -> int | None:
    data = get_json(SIZE_ENDPOINT, {"dataset": dataset, "config": config, "split": split})
    if "size" in data and isinstance(data["size"], dict):
        split_sizes = data["size"].get("splits", [])
        for item in split_sizes:
            if item.get("split") == split:
                return item.get("num_rows")
    if "num_rows" in data:
        return data["num_rows"]
    return None


def fetch_rows(dataset: str, offset: int, length: int, config: str = "default", split: str = "train"):
    data = get_json(
        ROWS_ENDPOINT,
        {
            "dataset": dataset,
            "config": config,
            "split": split,
            "offset": offset,
            "length": length,
        },
    )
    return [item["row"] for item in data.get("rows", [])]


def normalize_messages(row: dict):
    messages = row.get("messages")
    if isinstance(messages, list) and messages:
        return messages
    prompt = row.get("prompt")
    response = row.get("response") or row.get("completion") or row.get("answer")
    if prompt is not None and response is not None:
        return [
            {"role": "user", "content": str(prompt)},
            {"role": "assistant", "content": str(response)},
        ]
    return None


def record_from_row(row: dict, source_name: str):
    messages = normalize_messages(row)
    if not messages:
        return None
    return {
        "id": row.get("id"),
        "messages": messages,
        "source": source_name,
        "original_source": row.get("source"),
    }


def sample_standalone(source_name: str, dataset: str, n: int, rng: random.Random, sleep: float):
    if n <= 0:
        return []
    total_rows = get_num_rows(dataset) or n
    max_offset = max(total_rows - PAGE_SIZE, 0)
    offsets = list(range(0, max_offset + 1, PAGE_SIZE))
    rng.shuffle(offsets)

    records = []
    seen_ids = set()
    progress = tqdm(total=n, desc=source_name)
    for offset in offsets:
        rows = fetch_rows(dataset, offset=offset, length=PAGE_SIZE)
        rng.shuffle(rows)
        for row in rows:
            record = record_from_row(row, source_name)
            if record is None:
                continue
            record_id = record.get("id") or json.dumps(record["messages"], sort_keys=True)
            if record_id in seen_ids:
                continue
            seen_ids.add(record_id)
            records.append(record)
            progress.update(1)
            if len(records) >= n:
                progress.close()
                return records
        time.sleep(sleep)
    progress.close()
    return records


def looks_like_gsm_source(source: str | None):
    if not source:
        return False
    normalized = source.lower()
    return "gsm" in normalized and "persona" in normalized


def sample_gsm_from_mixture(n: int, rng: random.Random, max_scan: int, sleep: float):
    if n <= 0:
        return []
    records = []
    seen_ids = set()
    observed_sources = {}
    progress = tqdm(total=n, desc="gsm_from_tulu3_mixture")

    for offset in range(0, max_scan, PAGE_SIZE):
        rows = fetch_rows(MIXTURE_REPO, offset=offset, length=PAGE_SIZE)
        if not rows:
            break
        for row in rows:
            original_source = row.get("source")
            observed_sources[original_source] = observed_sources.get(original_source, 0) + 1
            if not looks_like_gsm_source(original_source):
                continue
            record = record_from_row(row, "gsm")
            if record is None:
                continue
            record_id = record.get("id") or json.dumps(record["messages"], sort_keys=True)
            if record_id in seen_ids:
                continue
            seen_ids.add(record_id)
            records.append(record)
            progress.update(1)
            if len(records) >= n:
                progress.close()
                return records
        time.sleep(sleep)

    progress.close()
    if len(records) < n:
        top_sources = sorted(observed_sources.items(), key=lambda item: item[1], reverse=True)[:20]
        print("Warning: did not collect requested GSM examples.")
        print("Observed mixture sources in scan:")
        for source, count in top_sources:
            print(f"  {source}: {count}")
    rng.shuffle(records)
    return records[:n]


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    records = []
    records.extend(
        sample_standalone(
            "algebra",
            STANDALONE_SOURCES["algebra"],
            args.algebra,
            rng,
            args.sleep,
        )
    )
    records.extend(
        sample_standalone(
            "instruction_following",
            STANDALONE_SOURCES["instruction_following"],
            args.instruction_following,
            rng,
            args.sleep,
        )
    )
    records.extend(sample_gsm_from_mixture(args.gsm, rng, args.max_mixture_scan, args.sleep))

    rng.shuffle(records)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    counts = {}
    for record in records:
        counts[record["source"]] = counts.get(record["source"], 0) + 1
    print(json.dumps({"output": str(output_path), "num_records": len(records), "counts": counts}, indent=2))


if __name__ == "__main__":
    main()

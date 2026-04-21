import argparse
import json
import os
import random
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


STANDALONE_SOURCES = {
    "algebra": "allenai/tulu-3-sft-personas-algebra",
    "code": "allenai/tulu-3-sft-personas-code",
    "math": "allenai/tulu-3-sft-personas-math",
    "math_grade": "allenai/tulu-3-sft-personas-math-grade",
    "instruction_following": "allenai/tulu-3-sft-personas-instruction-following",
}


def default_output_path() -> str:
    return "data/tulu3_reasoning/tulu3_reasoning_30k.jsonl"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=default_output_path())
    parser.add_argument("--math", type=int, default=8000)
    parser.add_argument("--algebra", type=int, default=4000)
    parser.add_argument("--math-grade", type=int, default=3000)
    parser.add_argument("--instruction-following", type=int, default=9000)
    parser.add_argument("--code", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--shuffle-buffer", type=int, default=10000)
    return parser.parse_args()


def normalize_messages(row: dict):
    messages = row.get("messages")
    if isinstance(messages, list) and messages:
        return messages
    prompt = row.get("prompt") or row.get("instruction") or row.get("input")
    response = row.get("response") or row.get("completion") or row.get("answer") or row.get("output")
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


def record_key(record: dict) -> str:
    return record.get("id") or json.dumps(record["messages"], sort_keys=True, ensure_ascii=False)


def sample_streaming_dataset(source_name: str, dataset: str, n: int, seed: int, shuffle_buffer: int):
    if n <= 0:
        return []

    try:
        dataset_iter = load_dataset(dataset, split="train", streaming=True)
    except Exception as exc:
        print(f"Warning: could not open {dataset} for {source_name}: {exc}")
        return None

    dataset_iter = dataset_iter.shuffle(seed=seed, buffer_size=shuffle_buffer)
    records = []
    seen = set()
    progress = tqdm(total=n, desc=source_name)

    for row in dataset_iter:
        record = record_from_row(row, source_name)
        if record is None:
            continue
        key = record_key(record)
        if key in seen:
            continue
        seen.add(key)
        records.append(record)
        progress.update(1)
        if len(records) >= n:
            break

    progress.close()
    if len(records) < n:
        print(f"Warning: requested {n} from {source_name}, collected {len(records)}.")
    return records


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    source_plan = [
        ("math", STANDALONE_SOURCES["math"], args.math),
        ("algebra", STANDALONE_SOURCES["algebra"], args.algebra),
        ("math_grade", STANDALONE_SOURCES["math_grade"], args.math_grade),
        ("instruction_following", STANDALONE_SOURCES["instruction_following"], args.instruction_following),
        ("code", STANDALONE_SOURCES["code"], args.code),
    ]

    for index, (source_name, dataset, count) in enumerate(source_plan):
        sampled = sample_streaming_dataset(
            source_name,
            dataset,
            count,
            seed=args.seed + index,
            shuffle_buffer=args.shuffle_buffer,
        )
        if sampled is None:
            sampled = []
        records.extend(sampled)

    rng.shuffle(records)

    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    counts = {}
    for record in records:
        counts[record["source"]] = counts.get(record["source"], 0) + 1

    metadata_path = output_path.with_suffix(".meta.json")
    metadata = {
        "output": str(output_path),
        "num_records": len(records),
        "counts": counts,
        "requested": {
            "math": args.math,
            "algebra": args.algebra,
            "math_grade": args.math_grade,
            "instruction_following": args.instruction_following,
            "code": args.code,
        },
        "seed": args.seed,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

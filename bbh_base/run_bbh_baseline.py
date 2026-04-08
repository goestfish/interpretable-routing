import argparse
import json
import os
import re
from pathlib import Path
from statistics import mean

import requests
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


BBH_TASKS = [
    "boolean_expressions",
    "causal_judgement",
    "date_understanding",
    "disambiguation_qa",
    "dyck_languages",
    "formal_fallacies",
    "geometric_shapes",
    "hyperbaton",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "logical_deduction_three_objects",
    "movie_recommendation",
    "multistep_arithmetic_two",
    "navigate",
    "object_counting",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "ruin_names",
    "salient_translation_error_detection",
    "snarks",
    "sports_understanding",
    "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
    "web_of_lies",
    "word_sorting",
]

RAW_BASE = "https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/main"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit-per-task", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--batch-size", type=int, default=1)
    return parser.parse_args()


def resolve_dtype(dtype_name: str):
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype_name]


def ensure_text(url: str, path: Path) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    path.write_text(response.text, encoding="utf-8")
    return response.text


def ensure_json(url: str, path: Path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    path.write_text(response.text, encoding="utf-8")
    return response.json()


def load_task_data(task: str, cache_dir: Path):
    prompt_url = f"{RAW_BASE}/cot-prompts/{task}.txt"
    data_url = f"{RAW_BASE}/bbh/{task}.json"
    prompt_path = cache_dir / "cot-prompts" / f"{task}.txt"
    data_path = cache_dir / "bbh" / f"{task}.json"
    prompt = ensure_text(prompt_url, prompt_path).strip()
    data = ensure_json(data_url, data_path)
    examples = data.get("examples", data)
    if not isinstance(examples, list):
        raise ValueError(f"Unexpected format for task {task}: top-level examples are missing.")
    return prompt, examples


def build_prompt(cot_prefix: str, question: str) -> str:
    cot_prefix = cot_prefix.rstrip()
    return f"{cot_prefix}\n\nQ: {question}\nA: Let's think step by step."


def extract_answer(text: str) -> str:
    cleaned = text.strip()
    patterns = [
        r"(?:[Tt]he answer is|[Ss]o the answer is|[Aa]nswer:)\s*(.+)",
        r"\b[Aa]:\s*(.+)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, cleaned, flags=re.DOTALL)
        if matches:
            candidate = matches[-1].strip()
            candidate = candidate.splitlines()[0].strip()
            candidate = re.split(r"(?<=[\.\!\?])\s", candidate)[0].strip()
            return candidate.strip(" .")
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    return lines[-1].strip(" .") if lines else ""


def normalize(text: str) -> str:
    text = text.strip().lower()
    text = text.replace("’", "'")
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" .,:;!?\"'")
    return text


def format_for_model(tokenizer, prompt: str):
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def generate_one(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float):
    rendered = format_for_model(tokenizer, prompt)
    inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
    do_sample = temperature > 0.0
    generation_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        generation_kwargs["temperature"] = temperature
    generation = model.generate(**generation_kwargs)
    new_tokens = generation[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    tasks = BBH_TASKS if args.tasks == "all" else [task.strip() for task in args.tasks.split(",") if task.strip()]

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    load_kwargs = dict(
        pretrained_model_name_or_path=args.model_dir,
        torch_dtype=resolve_dtype(args.dtype),
        trust_remote_code=True,
    )
    if args.device.startswith("cuda"):
        load_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
    if not args.device.startswith("cuda"):
        model.to(args.device)
    model.eval()

    predictions_path = output_dir / "predictions.jsonl"
    summary_path = output_dir / "summary.json"

    all_records = []
    per_task_scores = {}

    with predictions_path.open("w", encoding="utf-8") as pred_f:
        for task in tasks:
            cot_prompt, examples = load_task_data(task, cache_dir)
            if args.limit_per_task is not None:
                examples = examples[:args.limit_per_task]

            task_correct = 0
            task_records = []

            for example in tqdm(examples, desc=task):
                question = example["input"]
                target = example["target"]
                prompt = build_prompt(cot_prompt, question)
                completion = generate_one(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                )
                prediction = extract_answer(completion)
                correct = normalize(prediction) == normalize(target)

                record = {
                    "task": task,
                    "input": question,
                    "target": target,
                    "completion": completion,
                    "prediction": prediction,
                    "correct": correct,
                }
                pred_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                task_records.append(record)
                all_records.append(record)
                task_correct += int(correct)

            accuracy = task_correct / len(task_records) if task_records else 0.0
            per_task_scores[task] = {
                "num_examples": len(task_records),
                "accuracy": accuracy,
            }

    overall_accuracy = mean([r["correct"] for r in all_records]) if all_records else 0.0
    summary = {
        "model_dir": args.model_dir,
        "num_tasks": len(tasks),
        "num_examples": len(all_records),
        "overall_accuracy": overall_accuracy,
        "per_task": per_task_scores,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

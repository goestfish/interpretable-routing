import argparse
import json
import sys
from pathlib import Path
from statistics import mean

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "z_router"))

from z_router import install_z_router_blocks  # noqa: E402
from bbh_utils import BBH_TASKS, build_prompt, extract_answer, is_correct_prediction, load_task_data  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit-per-task", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--start-layer", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--num-z", type=int, default=None)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--soft-z", action="store_true")
    parser.add_argument("--sharing", default=None, choices=["independent", "cross_layer_shared"])
    parser.add_argument("--u-sharing", default=None, choices=["per_layer", "shared"])
    parser.add_argument("--sharing-group-size", type=int, default=None)
    parser.add_argument("--alpha-init", type=float, default=None)
    return parser.parse_args()


def resolve_dtype(dtype_name: str):
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype_name]


def format_for_model(tokenizer, prompt: str):
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
    new_tokens = generation[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def load_z_router_metadata(checkpoint_dir: Path):
    config_path = checkpoint_dir / "z_router_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing z-router config: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def install_and_load_z_router(model, checkpoint_dir: Path, args):
    metadata = load_z_router_metadata(checkpoint_dir)
    installed_layers = metadata.get("installed_layers")
    if not installed_layers:
        raise ValueError("Checkpoint metadata is missing installed_layers.")

    start_layer = args.start_layer if args.start_layer is not None else min(installed_layers)
    block_size = args.block_size if args.block_size is not None else len(installed_layers)
    num_z = args.num_z if args.num_z is not None else metadata["num_z"]
    tau = args.tau if args.tau is not None else metadata.get("tau", 1.0)
    soft_z = args.soft_z if args.soft_z else metadata.get("soft_z", False)
    allow_router_update = metadata.get("allow_router_update", False)
    sharing = args.sharing if args.sharing is not None else metadata.get("sharing", "independent")
    u_sharing = args.u_sharing if args.u_sharing is not None else metadata.get("u_sharing", "per_layer")
    sharing_group_size = (
        args.sharing_group_size if args.sharing_group_size is not None else metadata.get("sharing_group_size")
    )
    alpha_init = args.alpha_init if args.alpha_init is not None else metadata.get("alpha_init", 1e-3)

    install_z_router_blocks(
        model=model,
        start_layer=start_layer,
        block_size=block_size,
        num_z=num_z,
        tau=tau,
        hard=not soft_z,
        allow_router_update=allow_router_update,
        sharing=sharing,
        u_sharing=u_sharing,
        sharing_group_size=sharing_group_size,
        alpha_init=alpha_init,
    )

    state_path = checkpoint_dir / "z_router_trainable_state.pt"
    state = torch.load(state_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    remaining_unexpected = [name for name in unexpected if name in state]
    if remaining_unexpected:
        raise RuntimeError(f"Unexpected z-router checkpoint keys: {remaining_unexpected}")

    return {
        "checkpoint_dir": str(checkpoint_dir),
        "metadata": metadata,
        "loaded_keys": sorted(state.keys()),
        "missing_key_count": len(missing),
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    tasks = BBH_TASKS if args.tasks == "all" else [task.strip() for task in args.tasks.split(",") if task.strip()]

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    load_kwargs = dict(
        pretrained_model_name_or_path=args.model_dir,
        dtype=resolve_dtype(args.dtype),
        trust_remote_code=True,
    )
    if args.device.startswith("cuda"):
        load_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
    if not args.device.startswith("cuda"):
        model.to(args.device)

    z_router_info = install_and_load_z_router(model, checkpoint_dir, args)
    model.eval()

    predictions_path = output_dir / "predictions.jsonl"
    summary_path = output_dir / "summary.json"
    all_records = []
    per_task_scores = {}

    with predictions_path.open("w", encoding="utf-8") as pred_f:
        for task in tasks:
            cot_prompt, examples = load_task_data(task, cache_dir)
            if args.limit_per_task is not None:
                examples = examples[: args.limit_per_task]

            task_correct = 0
            task_records = []
            for example in tqdm(examples, desc=task):
                question = example["input"]
                target = example["target"]
                completion = generate_one(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=build_prompt(cot_prompt, question),
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                )
                prediction = extract_answer(task, completion, target)
                correct = is_correct_prediction(task, prediction, target)

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

            per_task_scores[task] = {
                "num_examples": len(task_records),
                "accuracy": task_correct / len(task_records) if task_records else 0.0,
            }

    summary = {
        "model_dir": args.model_dir,
        "z_router": z_router_info,
        "num_tasks": len(tasks),
        "num_examples": len(all_records),
        "overall_accuracy": mean([record["correct"] for record in all_records]) if all_records else 0.0,
        "per_task": per_task_scores,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

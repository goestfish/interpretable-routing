import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "z_router"))

from z_router import get_olmoe_layers, install_z_router_blocks  # noqa: E402
from bbh_utils import BBH_TASKS, build_prompt, load_task_data  # noqa: E402


class BBHPromptDataset(Dataset):
    def __init__(self, tasks, cache_dir: Path, limit_per_task: int | None = None):
        self.records = []
        for task in tasks:
            cot_prompt, examples = load_task_data(task, cache_dir)
            if limit_per_task is not None:
                examples = examples[:limit_per_task]
            for idx, example in enumerate(examples):
                self.records.append(
                    {
                        "task": task,
                        "index": idx,
                        "input": example["input"],
                        "target": example["target"],
                        "prompt": build_prompt(cot_prompt, example["input"]),
                    }
                )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--limit-per-task", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
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


def make_collate_fn(tokenizer, max_length: int):
    def collate(records):
        rendered = [format_for_model(tokenizer, record["prompt"]) for record in records]
        batch = tokenizer(
            rendered,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        return batch, records

    return collate


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

    installed = install_z_router_blocks(
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

    state = torch.load(checkpoint_dir / "z_router_trainable_state.pt", map_location="cpu")
    model.load_state_dict(state, strict=False)
    return metadata, installed, num_z


def entropy_from_counts(counts):
    total = sum(counts)
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counts:
        if count:
            p = count / total
            entropy -= p * math.log(p)
    return entropy


def update_stats(stats, task, layer, z_counts):
    layer_key = str(layer)
    task_stats = stats.setdefault(task, {}).setdefault(layer_key, {"num_tokens": 0, "z_counts": [0] * len(z_counts)})
    task_stats["num_tokens"] += sum(z_counts)
    task_stats["z_counts"] = [old + new for old, new in zip(task_stats["z_counts"], z_counts)]


def summarize_stats(stats, num_z):
    summary = {}
    for task, layer_map in stats.items():
        summary[task] = {}
        for layer, values in layer_map.items():
            counts = values["z_counts"]
            total = values["num_tokens"]
            distribution = [count / total if total else 0.0 for count in counts]
            dominant_z = max(range(num_z), key=lambda idx: counts[idx]) if counts else None
            entropy = entropy_from_counts(counts)
            summary[task][layer] = {
                "num_tokens": total,
                "z_counts": counts,
                "z_distribution": distribution,
                "dominant_z": dominant_z,
                "entropy": entropy,
                "normalized_entropy": entropy / math.log(num_z) if num_z > 1 else 0.0,
            }
    return summary




def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    tasks = BBH_TASKS if args.tasks == "all" else [task.strip() for task in args.tasks.split(",") if task.strip()]

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
    metadata, installed_layers, num_z = install_and_load_z_router(model, checkpoint_dir, args)
    model.eval()

    dataset = BBHPromptDataset(tasks=tasks, cache_dir=cache_dir, limit_per_task=args.limit_per_task)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=make_collate_fn(tokenizer, args.max_length),
    )

    layers = get_olmoe_layers(model)
    stats = {}
    example_path = output_dir / "z_usage_by_example.jsonl"

    with example_path.open("w", encoding="utf-8") as example_file:
        for batch, records in tqdm(dataloader, desc="z-usage"):
            batch = {key: value.to(model.device) for key, value in batch.items()}
            with torch.no_grad():
                model(**batch, use_cache=False)

            attention_mask = batch["attention_mask"].detach().cpu().bool()
            batch_size = attention_mask.shape[0]
            for row_idx in range(batch_size):
                record = records[row_idx]
                example_layers = {}
                valid_positions = attention_mask[row_idx]
                for layer_idx in installed_layers:
                    z_summary = layers[layer_idx].mlp.last_z_summary
                    if z_summary is None:
                        raise RuntimeError(f"Layer {layer_idx} did not record z usage.")
                    z = z_summary["z"].reshape(batch_size, -1, num_z)[row_idx]
                    z = z[valid_positions]
                    z_ids = z.argmax(dim=-1)
                    counts = torch.bincount(z_ids, minlength=num_z).tolist()
                    update_stats(stats, record["task"], layer_idx, counts)
                    example_layers[str(layer_idx)] = {
                        "num_tokens": int(sum(counts)),
                        "z_counts": counts,
                        "dominant_z": int(max(range(num_z), key=lambda idx: counts[idx])) if counts else None,
                    }
                example_file.write(
                    json.dumps(
                        {
                            "task": record["task"],
                            "index": record["index"],
                            "input": record["input"],
                            "target": record["target"],
                            "layers": example_layers,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    task_summary = summarize_stats(stats, num_z)
    report = {
        "model_dir": args.model_dir,
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_metadata": metadata,
        "installed_layers": installed_layers,
        "num_z": num_z,
        "num_examples": len(dataset),
        "limit_per_task": args.limit_per_task,
        "task_summary": task_summary,
    }
    (output_dir / "z_usage_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["checkpoint_dir", "installed_layers", "num_z", "num_examples"]}, indent=2))


if __name__ == "__main__":
    main()

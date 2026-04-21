import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from z_router import collect_z_router_losses, freeze_all_parameters, install_z_router_blocks, save_trainable_state


class JsonlMessagesDataset(Dataset):
    def __init__(self, path: str | Path):
        self.records = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start-layer", type=int, default=7)
    parser.add_argument("--block-size", type=int, default=2)
    parser.add_argument("--num-z", type=int, default=8)
    parser.add_argument("--z-mlp-hidden-size", type=int, default=None)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--soft-z", action="store_true")
    parser.add_argument("--allow-router-update", action="store_true")
    parser.add_argument("--sharing", default="independent", choices=["independent", "cross_layer_shared"])
    parser.add_argument("--u-sharing", default="per_layer", choices=["per_layer", "shared"])
    parser.add_argument("--sharing-group-size", type=int, default=None)
    parser.add_argument("--alpha-init", type=float, default=1e-3)
    parser.add_argument("--lambda-balance", type=float, default=0.0)
    parser.add_argument("--lambda-perturb", type=float, default=0.0)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--router-lr", type=float, default=1e-6)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--save-every-steps", type=int, default=200)
    return parser.parse_args()


def resolve_dtype(dtype_name: str):
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype_name]


def render_record(tokenizer, record):
    if "messages" in record:
        messages = record["messages"]
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return "\n".join(f"{message['role']}: {message['content']}" for message in messages)
    if "text" in record:
        return record["text"]
    if "prompt" in record and "response" in record:
        return f"User: {record['prompt']}\nAssistant: {record['response']}"
    raise ValueError("Each JSONL record must contain messages, text, or prompt/response.")


def make_collate_fn(tokenizer, max_length: int):
    def collate(records):
        texts = [render_record(tokenizer, record) for record in records]
        batch = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch

    return collate


def build_optimizer(model, lr: float, router_lr: float):
    z_params = []
    router_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if ".base_block.gate." in name:
            router_params.append(parameter)
        else:
            z_params.append(parameter)

    param_groups = []
    if z_params:
        param_groups.append({"params": z_params, "lr": lr})
    if router_params:
        param_groups.append({"params": router_params, "lr": router_lr})
    return torch.optim.AdamW(param_groups, weight_decay=0.0)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=resolve_dtype(args.dtype),
        device_map="auto",
        trust_remote_code=True,
    )
    model.train()
    model.config.use_cache = False

    freeze_all_parameters(model)
    installed_layers = install_z_router_blocks(
        model=model,
        start_layer=args.start_layer,
        block_size=args.block_size,
        num_z=args.num_z,
        z_mlp_hidden_size=args.z_mlp_hidden_size,
        tau=args.tau,
        hard=not args.soft_z,
        allow_router_update=args.allow_router_update,
        sharing=args.sharing,
        u_sharing=args.u_sharing,
        sharing_group_size=args.sharing_group_size,
        alpha_init=args.alpha_init,
    )

    dataset = JsonlMessagesDataset(args.train_jsonl)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=make_collate_fn(tokenizer, args.max_length),
    )
    optimizer = build_optimizer(model, lr=args.lr, router_lr=args.router_lr)

    global_step = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        progress = tqdm(dataloader, desc=f"epoch {epoch + 1}")
        for step, batch in enumerate(progress, start=1):
            batch = {key: value.to(model.device) for key, value in batch.items()}
            outputs = model(**batch)
            balance_loss, perturb_loss = collect_z_router_losses(model)
            raw_loss = outputs.loss + args.lambda_balance * balance_loss + args.lambda_perturb * perturb_loss
            loss = raw_loss / args.grad_accum_steps
            loss.backward()

            if step % args.grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                progress.set_postfix(
                    loss=float(outputs.loss.detach().cpu()),
                    balance=float(balance_loss.detach().cpu()),
                    perturb=float(perturb_loss.detach().cpu()),
                )

                if args.save_every_steps and global_step % args.save_every_steps == 0:
                    save_trainable_state(
                        model,
                        output_dir,
                        metadata={
                            "global_step": global_step,
                            "installed_layers": installed_layers,
                            "num_z": args.num_z,
                            "tau": args.tau,
                            "soft_z": args.soft_z,
                            "allow_router_update": args.allow_router_update,
                            "sharing": args.sharing,
                            "u_sharing": args.u_sharing,
                            "sharing_group_size": args.sharing_group_size,
                            "alpha_init": args.alpha_init,
                            "lambda_balance": args.lambda_balance,
                            "lambda_perturb": args.lambda_perturb,
                        },
                    )

    save_trainable_state(
        model,
        output_dir,
        metadata={
            "global_step": global_step,
            "installed_layers": installed_layers,
            "num_z": args.num_z,
            "tau": args.tau,
            "soft_z": args.soft_z,
            "allow_router_update": args.allow_router_update,
            "sharing": args.sharing,
            "u_sharing": args.u_sharing,
            "sharing_group_size": args.sharing_group_size,
            "alpha_init": args.alpha_init,
            "lambda_balance": args.lambda_balance,
            "lambda_perturb": args.lambda_perturb,
        },
    )


if __name__ == "__main__":
    main()

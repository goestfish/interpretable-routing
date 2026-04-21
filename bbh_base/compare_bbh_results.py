import argparse
import json
import math
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--z-dir", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--z-usage-dir", default=None)
    parser.add_argument("--report-output", default=None)
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def find_first_existing(directory: Path, names: list[str]) -> Path:
    for name in names:
        path = directory / name
        if path.exists():
            return path
    expected = ", ".join(names)
    raise FileNotFoundError(f"None of these files exist in {directory}: {expected}")


def load_predictions(path: Path):
    records = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            key = (record["task"], record["input"], record["target"])
            records[key] = record
    return records


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


def l1_distance(left, right):
    return sum(abs(a - b) for a, b in zip(left, right))


def build_z_usage_report(z_usage_summary, comparison, top_k=5):
    task_summary = z_usage_summary["task_summary"]
    num_z = z_usage_summary["num_z"]
    task_deltas = {row["task"]: row for row in comparison["task_deltas"]}
    layers = sorted({int(layer) for layer_map in task_summary.values() for layer in layer_map})
    report = {"headline": [], "layers": {}}

    for layer in layers:
        layer_key = str(layer)
        rows = []
        total_counts = [0] * num_z
        total_tokens = 0
        for task, layer_map in task_summary.items():
            if layer_key not in layer_map:
                continue
            values = layer_map[layer_key]
            counts = values["z_counts"]
            total_tokens += values["num_tokens"]
            total_counts = [old + new for old, new in zip(total_counts, counts)]
            rows.append(
                {
                    "task": task,
                    "dominant_z": values["dominant_z"],
                    "dominant_fraction": values["z_distribution"][values["dominant_z"]],
                    "normalized_entropy": values["normalized_entropy"],
                    "distribution": values["z_distribution"],
                }
            )

        overall_distribution = [count / total_tokens if total_tokens else 0.0 for count in total_counts]
        dominant_z = max(range(num_z), key=lambda idx: total_counts[idx])
        normalized_entropy = entropy_from_counts(total_counts) / math.log(num_z) if num_z > 1 else 0.0
        avg_task_entropy = sum(row["normalized_entropy"] for row in rows) / len(rows) if rows else 0.0

        report["layers"][layer_key] = {
            "overall_distribution": overall_distribution,
            "dominant_z": dominant_z,
            "dominant_fraction": overall_distribution[dominant_z],
            "normalized_entropy": normalized_entropy,
            "average_task_normalized_entropy": avg_task_entropy,
            "highest_entropy_tasks": sorted(rows, key=lambda row: row["normalized_entropy"], reverse=True)[:top_k],
            "lowest_entropy_tasks": sorted(rows, key=lambda row: row["normalized_entropy"])[:top_k],
            "most_distributionally_distinct_tasks": sorted(
                rows,
                key=lambda row: l1_distance(row["distribution"], overall_distribution),
                reverse=True,
            )[:top_k],
        }
        report["headline"].append(
            f"Layer {layer}: dominant z_{dominant_z} accounts for {overall_distribution[dominant_z]:.1%}; "
            f"average task normalized entropy is {avg_task_entropy:.3f}."
        )

    for key, title_rows in [
        ("largest_gains", sorted(task_deltas.values(), key=lambda row: row["delta_correct"], reverse=True)[:top_k]),
        ("largest_losses", sorted(task_deltas.values(), key=lambda row: row["delta_correct"])[:top_k]),
    ]:
        notes = []
        for row in title_rows:
            task = row["task"]
            note = {
                "task": task,
                "delta_correct": row["delta_correct"],
                "delta_accuracy": row["delta_accuracy"],
                "layers": {},
            }
            for layer in layers:
                layer_key = str(layer)
                values = task_summary.get(task, {}).get(layer_key)
                if values:
                    note["layers"][layer_key] = {
                        "dominant_z": values["dominant_z"],
                        "dominant_fraction": values["z_distribution"][values["dominant_z"]],
                        "normalized_entropy": values["normalized_entropy"],
                    }
            notes.append(note)
        report[key] = notes
    return report


def write_z_usage_markdown(path: Path, report, comparison):
    lines = ["# Z Usage Report", ""]
    lines.append("## Accuracy Context")
    lines.append("")
    lines.append(f"- Baseline accuracy: {comparison['baseline_overall_accuracy']:.4f}")
    lines.append(f"- Z-router accuracy: {comparison['z_overall_accuracy']:.4f}")
    lines.append(f"- Delta correct estimate: {comparison['delta_correct_estimate']}")
    lines.append(
        f"- Paired flips: +{comparison['paired_flips']['baseline_wrong_z_correct']} gains, "
        f"-{comparison['paired_flips']['baseline_correct_z_wrong']} losses"
    )
    lines.extend(["", "## Headline", ""])
    for item in report["headline"]:
        lines.append(f"- {item}")

    for layer, values in report["layers"].items():
        lines.extend(["", f"## Layer {layer}", ""])
        lines.append(
            f"- Dominant code: z_{values['dominant_z']} ({values['dominant_fraction']:.1%} of all analyzed tokens)"
        )
        lines.append(f"- Normalized entropy: {values['normalized_entropy']:.3f}")
        lines.append(f"- Average task normalized entropy: {values['average_task_normalized_entropy']:.3f}")
        lines.append(
            "- Overall distribution: "
            + ", ".join(f"z_{idx}={fraction:.1%}" for idx, fraction in enumerate(values["overall_distribution"]))
        )
        for title, key in [
            ("Highest-entropy tasks", "highest_entropy_tasks"),
            ("Lowest-entropy tasks", "lowest_entropy_tasks"),
            ("Most distributionally distinct tasks", "most_distributionally_distinct_tasks"),
        ]:
            lines.extend(["", f"{title}:"])
            for row in values[key]:
                lines.append(
                    f"- {row['task']}: H={row['normalized_entropy']:.3f}, "
                    f"dominant z_{row['dominant_z']}={row['dominant_fraction']:.1%}"
                )

    for title, key in [("Largest BBH Gains", "largest_gains"), ("Largest BBH Losses", "largest_losses")]:
        lines.extend(["", f"## {title}", ""])
        for row in report[key]:
            layer_bits = []
            for layer, values in row["layers"].items():
                layer_bits.append(
                    f"L{layer}: z_{values['dominant_z']}={values['dominant_fraction']:.1%}, H={values['normalized_entropy']:.3f}"
                )
            lines.append(
                f"- {row['task']}: delta_correct={row['delta_correct']}, "
                f"delta_accuracy={row['delta_accuracy']:.3f}; " + "; ".join(layer_bits)
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def maybe_write_combined_report(z_usage_dir: str | None, comparison, compare_output: str | None, report_output: str | None):
    if not z_usage_dir:
        return
    z_usage_dir = Path(z_usage_dir)
    summary_path = z_usage_dir / "z_usage_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing z usage summary: {summary_path}")
    z_usage_summary = load_json(summary_path)
    report = build_z_usage_report(z_usage_summary, comparison)
    if report_output:
        report_path = Path(report_output)
    elif compare_output:
        report_path = Path(compare_output).with_suffix(".md")
    else:
        report_path = Path("compare_bbh_report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_z_usage_markdown(report_path, report, comparison)
    report_json_path = report_path.with_suffix(".report.json")
    report_json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    args = parse_args()
    baseline_dir = Path(args.baseline_dir)
    z_dir = Path(args.z_dir)

    baseline_summary_path = find_first_existing(baseline_dir, ["summary.json"])
    z_summary_path = find_first_existing(z_dir, ["summary.json"])
    baseline_predictions_path = find_first_existing(baseline_dir, ["predictions.jsonl"])
    z_predictions_path = find_first_existing(z_dir, ["predictions.jsonl"])

    baseline = load_json(baseline_summary_path)
    z_summary = load_json(z_summary_path)
    baseline_predictions = load_predictions(baseline_predictions_path)
    z_predictions = load_predictions(z_predictions_path)

    task_rows = []
    for task, base_stats in baseline["per_task"].items():
        z_stats = z_summary["per_task"][task]
        n = base_stats["num_examples"]
        base_acc = base_stats["accuracy"]
        z_acc = z_stats["accuracy"]
        task_rows.append(
            {
                "task": task,
                "num_examples": n,
                "baseline_accuracy": base_acc,
                "z_accuracy": z_acc,
                "delta_accuracy": z_acc - base_acc,
                "delta_correct": round((z_acc - base_acc) * n),
            }
        )

    task_rows.sort(key=lambda row: row["delta_accuracy"], reverse=True)

    flips = {
        "baseline_wrong_z_correct": 0,
        "baseline_correct_z_wrong": 0,
        "both_correct": 0,
        "both_wrong": 0,
        "missing_pairs": 0,
    }
    flip_examples = {"gains": [], "losses": []}

    for key, base_record in baseline_predictions.items():
        z_record = z_predictions.get(key)
        if z_record is None:
            flips["missing_pairs"] += 1
            continue
        base_correct = bool(base_record["correct"])
        z_correct = bool(z_record["correct"])
        if not base_correct and z_correct:
            flips["baseline_wrong_z_correct"] += 1
            if len(flip_examples["gains"]) < 20:
                flip_examples["gains"].append(
                    {
                        "task": key[0],
                        "target": key[2],
                        "baseline_prediction": base_record["prediction"],
                        "z_prediction": z_record["prediction"],
                        "input": key[1],
                    }
                )
        elif base_correct and not z_correct:
            flips["baseline_correct_z_wrong"] += 1
            if len(flip_examples["losses"]) < 20:
                flip_examples["losses"].append(
                    {
                        "task": key[0],
                        "target": key[2],
                        "baseline_prediction": base_record["prediction"],
                        "z_prediction": z_record["prediction"],
                        "input": key[1],
                    }
                )
        elif base_correct and z_correct:
            flips["both_correct"] += 1
        else:
            flips["both_wrong"] += 1

    comparison = {
        "baseline_summary_path": str(baseline_summary_path),
        "z_summary_path": str(z_summary_path),
        "baseline_predictions_path": str(baseline_predictions_path),
        "z_predictions_path": str(z_predictions_path),
        "baseline_overall_accuracy": baseline["overall_accuracy"],
        "z_overall_accuracy": z_summary["overall_accuracy"],
        "delta_overall_accuracy": z_summary["overall_accuracy"] - baseline["overall_accuracy"],
        "num_examples": baseline["num_examples"],
        "delta_correct_estimate": round(
            (z_summary["overall_accuracy"] - baseline["overall_accuracy"]) * baseline["num_examples"]
        ),
        "task_deltas": task_rows,
        "paired_flips": flips,
        "flip_examples": flip_examples,
    }

    text = json.dumps(comparison, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    maybe_write_combined_report(args.z_usage_dir, comparison, args.output, args.report_output)


if __name__ == "__main__":
    main()

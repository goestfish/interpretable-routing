import json
import re
from pathlib import Path

import requests


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

CHOICE_TASKS = {
    "date_understanding",
    "disambiguation_qa",
    "geometric_shapes",
    "hyperbaton",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "logical_deduction_three_objects",
    "movie_recommendation",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "ruin_names",
    "salient_translation_error_detection",
    "snarks",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
}

BOOLEAN_TASKS = {
    "boolean_expressions",
    "formal_fallacies",
    "navigate",
    "sports_understanding",
    "web_of_lies",
}

RAW_BASE = "https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/main"


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


def normalize_text(text: str) -> str:
    text = str(text).strip().lower()
    text = text.replace("’", "'")
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .,:;!?\"'")


def normalize_boolean(text: str) -> str | None:
    normalized = normalize_text(text)
    if re.search(r"\btrue\b", normalized):
        return "true"
    if re.search(r"\bfalse\b", normalized):
        return "false"
    if re.search(r"\byes\b", normalized):
        return "yes"
    if re.search(r"\bno\b", normalized):
        return "no"
    return None


def normalize_choice(text: str) -> str | None:
    normalized = normalize_text(text)
    patterns = [
        r"(?:^|\b)(?:option|choice|answer)\s*[:\-]?\s*\(?([a-z])\)?(?:\b|$)",
        r"^\(?([a-z])\)?$",
        r"(?:^|\s)\(([a-z])\)(?:\s|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match:
            return match.group(1)
    return None


def normalize_number(text: str) -> str | None:
    normalized = normalize_text(text)
    matches = re.findall(r"[-+]?\d+(?:\.\d+)?", normalized.replace(",", ""))
    if not matches:
        return None
    number = matches[-1]
    if number.endswith(".0"):
        number = number[:-2]
    return number


def candidate_answer_spans(text: str):
    cleaned = str(text).strip()
    spans = []
    patterns = [
        r"(?:[Tt]he answer is|[Ss]o the answer is|[Ff]inal answer is|[Aa]nswer:|[Aa]nswer is|[Tt]herefore,? the answer is)\s*(.+)",
        r"(?:[Tt]he final result is|[Ss]o,? the final result is)\s*:?\s*(.+)",
        r"\b[Aa]:\s*(.+)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, cleaned, flags=re.DOTALL)
        for match in matches:
            spans.append(match.strip())

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    spans.extend(reversed(lines[-3:]))
    spans.append(cleaned)

    shortened = []
    for span in spans:
        shortened.append(span)
        shortened.extend(re.split(r"(?<=[\.\!\?])\s+", span)[:2])
        if ":" in span:
            shortened.append(span.split(":")[-1].strip())
    return [span for span in shortened if span]


def canonical_target(task: str, target: str) -> str:
    if task in BOOLEAN_TASKS:
        boolean = normalize_boolean(target)
        if boolean is not None:
            return boolean
    if task in CHOICE_TASKS:
        choice = normalize_choice(target)
        if choice is not None:
            return choice
    number = normalize_number(target)
    if number is not None and normalize_text(target) == number:
        return number
    return normalize_text(target)


def extract_answer(task: str, completion: str, target: str | None = None) -> str:
    target_canonical = canonical_target(task, target) if target is not None else ""
    spans = candidate_answer_spans(completion)

    if task in BOOLEAN_TASKS or target_canonical in {"true", "false", "yes", "no"}:
        for span in spans:
            boolean = normalize_boolean(span)
            if boolean is not None:
                return boolean

    if task in CHOICE_TASKS or re.fullmatch(r"[a-z]", target_canonical):
        for span in spans:
            choice = normalize_choice(span)
            if choice is not None:
                return choice

    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", target_canonical):
        for span in spans:
            number = normalize_number(span)
            if number is not None:
                return number

    for span in spans:
        normalized = normalize_text(span)
        if target_canonical and target_canonical in normalized:
            return target_canonical
    return normalize_text(spans[0]) if spans else ""


def is_correct_prediction(task: str, prediction: str, target: str) -> bool:
    return normalize_text(prediction) == normalize_text(canonical_target(task, target))


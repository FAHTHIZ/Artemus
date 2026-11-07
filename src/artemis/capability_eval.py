"""Held-out math and Python diagnostics; generated code is parsed, never executed."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import torch

from artemis.config import ModelConfig
from artemis.generate import load_tokenizer
from artemis.model import ArtemisLM
from artemis.train import choose_device

_NUMBER = re.compile(r"^\s*\$?\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*%?\s*[.!]?\s*$")
_ANSWER_LABEL = re.compile(
    r"^(?:(?:the\s+)?final(?:\s+answer)?|(?:the\s+)?answer)\s*(?::|=|\bis\b)\s*(.+)$",
    flags=re.IGNORECASE,
)


def normalize_number(candidate: str) -> str | None:
    """Normalize one numeric answer, without guessing from arbitrary prose."""
    match = _NUMBER.fullmatch(candidate.strip())
    if not match:
        return None
    try:
        value = Decimal(match.group(1).replace(",", ""))
    except InvalidOperation:
        return None
    if not value.is_finite():
        return None
    normalized = format(value.normalize(), "f")
    return "0" if normalized in {"-0", "-0.0"} else normalized


def extract_numeric_answer(text: str) -> str | None:
    """Extract a clearly marked final numeric answer."""
    marked = re.findall(r"####\s*([^\r\n]+)", text)
    if marked:
        return normalize_number(marked[-1])
    for line in reversed(text.strip().splitlines()):
        candidate = line.strip()
        label = _ANSWER_LABEL.match(candidate)
        if label:
            candidate = label.group(1).strip()
        normalized = normalize_number(candidate)
        if normalized is not None:
            return normalized
    return None


def strip_code_fence(text: str) -> str:
    value = text.strip()
    if not value.startswith(chr(96) * 3):
        return value
    lines = value.splitlines()
    if len(lines) >= 2 and lines[-1].strip() == chr(96) * 3:
        return "\n".join(lines[1:-1]).strip()
    return "\n".join(lines[1:]).strip()


def is_python_syntax_valid(source: str) -> bool:
    """Parse Python syntax only. This function never executes generated code."""
    code = strip_code_fence(source)
    if not code:
        return False
    try:
        ast.parse(code)
    except (SyntaxError, ValueError, TypeError):
        return False
    return True


def load_evaluation_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid evaluation JSONL at line {line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Evaluation record at line {line_number} must be an object")
            required = ("id", "task_type", "prompt", "target")
            if any(not isinstance(row.get(key), str) or not row[key].strip() for key in required):
                raise ValueError(f"Evaluation record at line {line_number} is missing required text fields")
            if row["task_type"] not in {"math", "python"}:
                raise ValueError(f"Unsupported task_type at line {line_number}")
            if row["id"] in seen:
                raise ValueError(f"Duplicate evaluation ID: {row['id']}")
            if row["task_type"] == "math" and normalize_number(str(row.get("expected_answer", ""))) is None:
                raise ValueError(f"Math evaluation record {row['id']} has no parseable expected_answer")
            seen.add(row["id"])
            records.append(row)
    if not records:
        raise ValueError("Evaluation dataset contains no records")
    return records


@torch.inference_mode()
def run_capability_evaluation(
    *,
    checkpoint: Path,
    data_path: Path,
    output_dir: Path,
    max_new_tokens: int = 96,
    max_math_examples: int = 0,
    max_code_examples: int = 0,
    seed: int = 1337,
    device_name: str = "auto",
) -> dict[str, Any]:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    if max_new_tokens < 1 or max_math_examples < 0 or max_code_examples < 0:
        raise ValueError("Generation limit must be positive and example limits non-negative")

    all_records = load_evaluation_records(data_path)
    math_records = [row for row in all_records if row["task_type"] == "math"]
    python_records = [row for row in all_records if row["task_type"] == "python"]
    if max_math_examples:
        math_records = math_records[:max_math_examples]
    if max_code_examples:
        python_records = python_records[:max_code_examples]
    records = math_records + python_records
    if not records:
        raise ValueError("No evaluation records remain after applying task limits")

    device = choose_device(device_name)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    if payload.get("format_version") not in {1, 2, 3}:
        raise ValueError("Unsupported checkpoint format")
    tokenizer = load_tokenizer(payload)
    config = ModelConfig.from_dict(payload["model_config"])
    if config.vocab_size != tokenizer.vocab_size:
        raise ValueError("Checkpoint model and tokenizer vocabulary sizes do not match")
    model = ArtemisLM(config).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    data_sha = hashlib.sha256(data_path.read_bytes()).hexdigest()
    result_path = output_dir / "results.jsonl"
    math_correct = math_evaluated = math_unparsed = 0
    python_syntax_valid = python_exact_match = python_evaluated = 0
    skipped_context = 0
    with result_path.open("w", encoding="utf-8") as output:
        for row in records:
            prompt = f"<|prompter|>\n{row['prompt'].strip()}\n<|assistant|>\n"
            prompt_ids = tokenizer.encode(prompt)
            if len(prompt_ids) > config.block_size:
                skipped_context += 1
                result = {
                    "id": row["id"],
                    "task_type": row["task_type"],
                    "status": "skipped_context",
                    "prompt_tokens": len(prompt_ids),
                    "context_limit": config.block_size,
                }
                output.write(json.dumps(result, ensure_ascii=False) + "\n")
                continue

            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            generated = model.generate(
                input_ids,
                max_new_tokens,
                temperature=1.0,
                top_k=1,
                eos_token_id=tokenizer.stop_token_id,
                use_cache=True,
            )[0].tolist()
            response_ids = generated[len(prompt_ids):]
            response = tokenizer.decode(response_ids)
            if "<|end|>" in response:
                response = response.split("<|end|>", 1)[0]
            response = response.strip()
            result: dict[str, Any] = {
                "id": row["id"],
                "source": row.get("source"),
                "task_type": row["task_type"],
                "status": "evaluated",
                "prompt_tokens": len(prompt_ids),
                "generated_tokens": len(response_ids),
                "response": response,
            }
            if row["task_type"] == "math":
                expected = normalize_number(str(row["expected_answer"]))
                predicted = extract_numeric_answer(response)
                correct = predicted is not None and predicted == expected
                math_evaluated += 1
                math_correct += int(correct)
                math_unparsed += int(predicted is None)
                result.update({
                    "expected_answer": expected,
                    "predicted_answer": predicted,
                    "exact_match": correct,
                })
            else:
                expected_code = strip_code_fence(str(row["target"]))
                predicted_code = strip_code_fence(response)
                syntax_valid = is_python_syntax_valid(response)
                exact_match = bool(predicted_code) and predicted_code == expected_code
                python_evaluated += 1
                python_syntax_valid += int(syntax_valid)
                python_exact_match += int(exact_match)
                result.update({
                    "syntax_valid": syntax_valid,
                    "exact_text_match": exact_match,
                    "execution_tests_run": 0,
                })
            output.write(json.dumps(result, ensure_ascii=False) + "\n")

    def rate(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    summary = {
        "schema_version": 1,
        "checkpoint_sha256": checkpoint_sha,
        "evaluation_data_sha256": data_sha,
        "checkpoint_step": payload.get("step"),
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(device),
        "seed": seed,
        "generation": {"max_new_tokens": max_new_tokens, "top_k": 1, "temperature": 1.0},
        "math": {
            "evaluated": math_evaluated,
            "correct": math_correct,
            "unparsed_answers": math_unparsed,
            "exact_match_accuracy": rate(math_correct, math_evaluated),
        },
        "python": {
            "evaluated": python_evaluated,
            "syntax_valid": python_syntax_valid,
            "exact_text_matches": python_exact_match,
            "syntax_valid_rate": rate(python_syntax_valid, python_evaluated),
            "exact_text_match_rate": rate(python_exact_match, python_evaluated),
            "execution_tests_run": 0,
            "note": "Syntax and exact text only; generated code is never executed by this evaluator.",
        },
        "skipped_context": skipped_context,
        "total_records_selected": len(records),
        "result_file": str(result_path),
        "evaluation_scope": (
            "Internal held-out partitions from the declared source training splits. "
            "These diagnostic metrics are not official GSM8K or MBPP benchmark scores."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/capability-eval"))
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--max-math-examples", type=int, default=0)
    parser.add_argument("--max-code-examples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_capability_evaluation(
        checkpoint=args.checkpoint,
        data_path=args.data,
        output_dir=args.output_dir,
        max_new_tokens=args.max_new_tokens,
        max_math_examples=args.max_math_examples,
        max_code_examples=args.max_code_examples,
        seed=args.seed,
        device_name=args.device,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

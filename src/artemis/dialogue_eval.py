"""Run reproducible, scripted multi-turn conversations against an Artemis checkpoint.

The generated conversations are evaluation traces only. They are not silently
recycled into training data; any future curation requires explicit review.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from artemis.config import ModelConfig
from artemis.generate import load_tokenizer
from artemis.model import ArtemisLM
from artemis.train import choose_device


def build_dialogue_prompt(history: list[dict[str, str]], user_text: str) -> str:
    if not isinstance(user_text, str) or not user_text.strip():
        raise ValueError("user_text must be a non-empty string")
    parts: list[str] = []
    for message in history:
        role = message.get("role")
        content = message.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            raise ValueError("history messages must have user/assistant role and string content")
        marker = "prompter" if role == "user" else "assistant"
        parts.append(f"<|{marker}|>\n{content}\n<|end|>\n")
    parts.append(f"<|prompter|>\n{user_text.strip()}\n<|assistant|>\n")
    return "".join(parts)


def fit_dialogue_context(
    tokenizer: Any,
    history: list[dict[str, str]],
    user_text: str,
    block_size: int,
) -> tuple[str, list[int], int]:
    """Fit a dialogue prompt by dropping the oldest complete turns first.

    The current user turn is never silently truncated. If it cannot fit on its
    own, fail with an explicit error rather than feeding an invalid context.
    """
    if block_size < 1:
        raise ValueError("block_size must be positive")
    retained_history = list(history)
    dropped_messages = 0
    while True:
        prompt = build_dialogue_prompt(retained_history, user_text)
        prompt_ids = tokenizer.encode(prompt)
        if len(prompt_ids) <= block_size:
            return prompt, prompt_ids, dropped_messages
        if not retained_history:
            raise ValueError(
                f"current user turn uses {len(prompt_ids)} tokens after history trimming; "
                f"context window is {block_size}"
            )
        drop_count = min(2, len(retained_history))
        del retained_history[:drop_count]
        dropped_messages += drop_count


def extract_assistant_response(
    tokenizer: Any,
    generated_token_ids: list[int],
) -> tuple[str, bool]:
    ended_turn = tokenizer.stop_token_id in generated_token_ids
    text = tokenizer.decode(generated_token_ids)
    if "<|end|>" in text:
        text = text.split("<|end|>", 1)[0]
    return text.strip(), ended_turn


def load_scenarios(path: Path) -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Scenario at line {line_number} must be a JSON object")
            scenario_id = row.get("id")
            capability = row.get("capability")
            turns = row.get("turns")
            if not isinstance(scenario_id, str) or not scenario_id:
                raise ValueError(f"Scenario at line {line_number} has no id")
            if not isinstance(capability, str) or not capability:
                raise ValueError(f"Scenario {scenario_id} has no capability label")
            if not isinstance(turns, list) or not turns or not all(
                isinstance(turn, str) and turn.strip() for turn in turns
            ):
                raise ValueError(f"Scenario {scenario_id} must contain non-empty user turns")
            scenarios.append(row)
    if not scenarios:
        raise ValueError("scenario file contains no scenarios")
    ids = [row["id"] for row in scenarios]
    if len(ids) != len(set(ids)):
        raise ValueError("scenario IDs must be unique")
    return scenarios


@torch.inference_mode()
def run_dialogues(
    *,
    checkpoint: Path,
    scenarios_path: Path,
    output_dir: Path,
    max_new_tokens: int = 64,
    top_k: int = 1,
    temperature: float = 1.0,
    seed: int = 1337,
    device_name: str = "cpu",
) -> dict[str, Any]:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    scenarios = load_scenarios(scenarios_path)
    device = choose_device(device_name)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    if payload.get("format_version") not in {1, 2, 3}:
        raise ValueError("unsupported checkpoint format")
    tokenizer = load_tokenizer(payload)
    config = ModelConfig.from_dict(payload["model_config"])
    if config.vocab_size != tokenizer.vocab_size:
        raise ValueError("checkpoint tokenizer and model vocabulary do not match")
    model = ArtemisLM(config).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = output_dir / "dialogues.jsonl"
    transcript_records: list[dict[str, Any]] = []
    total_turns = 0
    nonempty_turns = 0
    ended_turns = 0
    context_trimmed_turn_count = 0
    history_messages_dropped = 0

    with transcript_path.open("w", encoding="utf-8") as output:
        for scenario in scenarios:
            history: list[dict[str, str]] = []
            turn_records: list[dict[str, Any]] = []
            for turn_index, user_text in enumerate(scenario["turns"], start=1):
                prompt, prompt_ids, dropped_messages = fit_dialogue_context(
                    tokenizer, history, user_text, config.block_size
                )
                context_trimmed_turn_count += int(dropped_messages > 0)
                history_messages_dropped += dropped_messages
                input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
                generated = model.generate(
                    input_ids,
                    max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    eos_token_id=tokenizer.stop_token_id,
                    use_cache=True,
                )[0].tolist()
                new_ids = generated[len(prompt_ids):]
                response, ended = extract_assistant_response(tokenizer, new_ids)
                total_turns += 1
                nonempty_turns += int(bool(response))
                ended_turns += int(ended)
                turn_records.append({
                    "turn": turn_index,
                    "user": user_text,
                    "assistant": response,
                    "prompt_tokens": len(prompt_ids),
                    "history_messages_dropped": dropped_messages,
                    "generated_tokens": len(new_ids),
                    "ended_with_turn_marker": ended,
                    "nonempty_response": bool(response),
                })
                history.append({"role": "user", "content": user_text})
                history.append({"role": "assistant", "content": response})

            record = {
                "scenario_id": scenario["id"],
                "capability": scenario["capability"],
                "checkpoint_sha256": checkpoint_hash,
                "checkpoint_step": payload.get("step"),
                "model_parameters": sum(p.numel() for p in model.parameters()),
                "turns": turn_records,
            }
            transcript_records.append(record)
            output.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_step": payload.get("step"),
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "device": str(device),
        "scenario_count": len(scenarios),
        "turn_count": total_turns,
        "nonempty_response_count": nonempty_turns,
        "turn_marker_count": ended_turns,
        "empty_response_count": total_turns - nonempty_turns,
        "context_trimmed_turn_count": context_trimmed_turn_count,
        "history_messages_dropped": history_messages_dropped,
        "context_overflow_count": 0,
        "transcript_file": str(transcript_path),
        "evaluation_note": (
            "This is a multi-turn interaction trace and basic generation-health check, "
            "not a correctness, reasoning, safety, or capability benchmark. Generated "
            "responses are not added to training automatically."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scenarios", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/dialogue-eval"))
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_dialogues(
        checkpoint=args.checkpoint,
        scenarios_path=args.scenarios,
        output_dir=args.output_dir,
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k,
        temperature=args.temperature,
        seed=args.seed,
        device_name=args.device,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

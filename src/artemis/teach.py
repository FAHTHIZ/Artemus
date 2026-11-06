"""Interactive human-teaching loop for ArtemusLM.

This captures reviewed examples as candidate SFT data. It never updates weights
or feeds unreviewed model outputs back into training automatically.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import torch

from artemis.config import ModelConfig
from artemis.dialogue_eval import extract_assistant_response, fit_dialogue_context
from artemis.generate import load_tokenizer
from artemis.model import ArtemisLM
from artemis.train import choose_device


def format_sft_text(
    prompt: str,
    answer: str,
    history: list[dict[str, str]] | None = None,
) -> str:
    """Format a reviewed answer with the prior user/assistant turns that informed it."""
    if not prompt.strip() or not answer.strip():
        raise ValueError("prompt and answer must not be empty")
    segments: list[str] = []
    for message in history or []:
        role = message.get("role")
        content = message.get("content", "")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            raise ValueError("history messages must have user/assistant roles and string content")
        if not content.strip():
            continue
        marker = "<|prompter|>" if role == "user" else "<|assistant|>"
        segments.append(f"{marker}\n{content.strip()}\n<|end|>")
    segments.append(
        "<|prompter|>\n"
        + prompt.strip()
        + "\n<|assistant|>\n"
        + answer.strip()
        + "\n<|end|>"
    )
    return "\n".join(segments)




def write_candidate(
    output_path: Path,
    *,
    prompt: str,
    answer: str,
    session_id: str,
    turn: int,
    source_type: str,
    checkpoint_sha256: str,
    history: list[dict[str, str]] | None = None,
) -> None:
    if source_type not in {"human_correction", "human_approved_model_output"}:
        raise ValueError("unsupported teaching source type")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "text": format_sft_text(prompt, answer, history),
        "source": "interactive_human_teaching",
        "source_type": source_type,
        "human_reviewed": True,
        "session_id": session_id,
        "turn": turn,
        "context_turns": len(history or []),
        "checkpoint_sha256": checkpoint_sha256,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with output_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/human-teaching/candidates.jsonl"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--allow-human-approved-model-output",
        action="store_true",
        help="permit explicitly approved model responses to become SFT candidates; off by default",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint does not exist: {args.checkpoint}")
    if args.max_new_tokens < 1 or args.top_k < 1 or args.temperature <= 0:
        raise SystemExit("max-new-tokens and top-k must be positive; temperature must be > 0")

    device = choose_device(args.device)
    torch.manual_seed(args.seed)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if payload.get("format_version") not in {1, 2, 3}:
        raise SystemExit("Unsupported checkpoint format")

    tokenizer = load_tokenizer(payload)
    config = ModelConfig.from_dict(payload["model_config"])
    if config.vocab_size != tokenizer.vocab_size:
        raise SystemExit("Checkpoint tokenizer/model vocabulary mismatch")
    model = ArtemisLM(config).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    import hashlib

    checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    session_id = uuid4().hex
    history: list[dict[str, str]] = []
    turn = 0

    print("Artemus human-teaching session")
    print("Type /quit to exit. After each answer, choose correct, approve, or skip.")
    if not args.allow_human_approved_model_output:
        print("Model answers are NOT saved unless you provide a human correction.")

    while True:
        prompt = input("\nYou: ").strip()
        if not prompt:
            continue
        if prompt.lower() in {"/quit", "/exit"}:
            break
        if prompt.startswith("/clear"):
            history.clear()
            print("Conversation context cleared.")
            continue

        turn += 1
        full_prompt, prompt_ids, dropped = fit_dialogue_context(
            tokenizer, history, prompt, config.block_size
        )
        if dropped:
            print(f"[Context manager trimmed {dropped} oldest messages.]")
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        with torch.inference_mode():
            generated = model.generate(
                input_ids,
                args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                eos_token_id=tokenizer.stop_token_id,
                use_cache=True,
            )[0].tolist()
        answer, _ = extract_assistant_response(tokenizer, generated[len(prompt_ids):])
        print(f"\nArtemus: {answer or '[empty response]'}")

        choice = input("\nTeaching action [c=correct, a=approve model output, s=skip, q=quit]: ").strip().lower()
        selected_answer = ""
        source_type = ""
        if choice == "c":
            selected_answer = input("Your corrected / preferred answer: ").strip()
            source_type = "human_correction"
        elif choice == "a":
            if args.allow_human_approved_model_output:
                selected_answer = answer
                source_type = "human_approved_model_output"
            else:
                print(
                    "Not saved: rerun with --allow-human-approved-model-output "
                    "to include approved model generations."
                )
        elif choice in {"q", "quit"}:
            break
        elif choice == "s":
            pass
        else:
            print("Unknown choice; this turn was not added to the teaching set.")

        if selected_answer.strip():
            write_candidate(
                args.output,
                prompt=prompt,
                answer=selected_answer,
                session_id=session_id,
                turn=turn,
                source_type=source_type,
                checkpoint_sha256=checkpoint_hash,
                history=history,
            )
            print(f"[Saved reviewed candidate to {args.output}]")

        history.extend(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": selected_answer or answer},
            ]
        )

    print(f"Session complete. Candidate data: {args.output}")


if __name__ == "__main__":
    main()

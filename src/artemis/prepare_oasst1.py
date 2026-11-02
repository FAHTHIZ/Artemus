"""Prepare a small, auditable OASST1 conversation corpus for Artemis.

The upstream archive is downloaded by the CI workflow, not committed to Git.
Only reviewed English, non-deleted, non-synthetic messages are retained.
Conversation trees are split as groups to prevent the same tree crossing splits.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

SOURCE_REVISION = "a4992e1595393714c9e6c567b1ed645bbf7d00ee"
SOURCE_URL = (
    "https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/"
    + SOURCE_REVISION
    + "/2023-04-12_oasst_ready.messages.jsonl.gz"
)
EXPECTED_SHA256 = "286a6e9a5a413b3272ae9c0b5a20d327983dea1c24342ae28cb244a6da65185c"
DATASET_LICENSE = "Apache-2.0"


def is_eligible(message: dict[str, Any]) -> bool:
    """Return true only for reviewed, non-deleted English human messages."""
    if message.get("lang") != "en":
        return False
    if message.get("review_result") is not True or message.get("deleted") is not False:
        return False
    if message.get("synthetic") is True:
        return False
    if message.get("role") not in {"prompter", "assistant"}:
        return False
    text = message.get("text")
    if not isinstance(text, str) or not text.strip():
        return False

    labels = message.get("labels") or {}
    pii_label = labels.get("pii")
    if isinstance(pii_label, dict) and float(pii_label.get("value", 0.0)) > 0.0:
        return False
    return True


def _conversation_for(
    assistant: dict[str, Any],
    by_id: dict[str, dict[str, Any]],
    max_messages: int,
    max_message_chars: int,
) -> str | None:
    chain: list[dict[str, Any]] = []
    current: dict[str, Any] | None = assistant
    seen: set[str] = set()

    while current is not None and len(chain) < max_messages:
        message_id = str(current.get("message_id", ""))
        if not message_id or message_id in seen:
            break
        seen.add(message_id)
        chain.append(current)
        parent_id = current.get("parent_id")
        current = by_id.get(str(parent_id)) if parent_id else None

    chain.reverse()
    while chain and chain[0].get("role") != "prompter":
        chain.pop(0)

    if len(chain) < 2 or chain[-1].get("role") != "assistant":
        return None
    roles = [item.get("role") for item in chain]
    if any(left == right for left, right in zip(roles, roles[1:])):
        return None

    parts: list[str] = []
    for item in chain:
        body = str(item["text"]).strip()
        if len(body) > max_message_chars:
            return None
        role = "prompter" if item["role"] == "prompter" else "assistant"
        parts.append(f"<|{role}|>\n{body}\n")
    parts.append("<|end|>\n")
    return "".join(parts)


def build_examples(
    messages: Iterable[dict[str, Any]],
    *,
    max_examples: int = 15000,
    max_messages: int = 8,
    max_message_chars: int = 4000,
) -> tuple[list[tuple[str, str]], int]:
    """Return (tree_id, conversation) examples and eligible message count."""
    eligible = [message for message in messages if is_eligible(message)]
    by_id = {
        str(message["message_id"]): message
        for message in eligible
        if message.get("message_id")
    }

    examples: dict[str, tuple[str, str]] = {}
    for message in eligible:
        if message.get("role") != "assistant":
            continue
        conversation = _conversation_for(
            message, by_id, max_messages=max_messages,
            max_message_chars=max_message_chars,
        )
        if conversation is None:
            continue
        tree_id = str(message.get("message_tree_id") or message.get("message_id"))
        sample_id = str(message["message_id"])
        key = hashlib.sha256((tree_id + ":" + sample_id).encode("utf-8")).hexdigest()
        examples[key] = (tree_id, conversation)

    ordered = sorted(examples.items(), key=lambda item: item[0])[:max_examples]
    return [item[1] for item in ordered], len(eligible)


def load_jsonl_gzip(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {line_number}") from exc
    return rows


def prepare(
    source_path: Path,
    output_dir: Path,
    *,
    expected_sha256: str = EXPECTED_SHA256,
    max_examples: int = 15000,
) -> dict[str, Any]:
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if expected_sha256 and digest != expected_sha256:
        raise ValueError(
            f"Dataset SHA-256 mismatch: expected {expected_sha256}, got {digest}"
        )

    messages = load_jsonl_gzip(source_path)
    examples, eligible_count = build_examples(messages, max_examples=max_examples)
    if len(examples) < 20:
        raise ValueError(f"Only {len(examples)} usable conversation examples were found")

    train_parts: list[str] = []
    validation_parts: list[str] = []
    train_trees: set[str] = set()
    validation_trees: set[str] = set()
    for tree_id, conversation in examples:
        bucket = int(hashlib.sha256(tree_id.encode("utf-8")).hexdigest()[:8], 16) % 10
        if bucket == 0:
            validation_parts.append(conversation)
            validation_trees.add(tree_id)
        else:
            train_parts.append(conversation)
            train_trees.add(tree_id)

    if not train_parts or not validation_parts:
        raise ValueError("Tree-level split produced an empty train or validation set")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train.txt").write_text("\n".join(train_parts), encoding="utf-8")
    (output_dir / "validation.txt").write_text(
        "\n".join(validation_parts), encoding="utf-8"
    )

    manifest = {
        "dataset": "OpenAssistant Conversations (OASST1), ready-for-export messages",
        "license": DATASET_LICENSE,
        "source_url": SOURCE_URL,
        "source_revision": SOURCE_REVISION,
        "source_sha256": digest,
        "filter": {
            "language": "en",
            "review_result": True,
            "deleted": False,
            "synthetic": False,
            "positive_pii_label": "excluded",
        },
        "eligible_messages": eligible_count,
        "conversation_examples": len(examples),
        "train_examples": len(train_parts),
        "validation_examples": len(validation_parts),
        "train_tree_groups": len(train_trees),
        "validation_tree_groups": len(validation_trees),
        "split": "deterministic 90/10 by message_tree_id hash; no tree crosses splits",
        "tokenizer_note": (
            "Artemus baseline UTF-8 byte tokenizer; preprocessing does not imply "
            "the current model is instruction-tuned or capable."
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="OASST1 .jsonl.gz file")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/oasst1"))
    parser.add_argument("--max-examples", type=int, default=15000)
    args = parser.parse_args()
    manifest = prepare(args.source, args.output_dir, max_examples=args.max_examples)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

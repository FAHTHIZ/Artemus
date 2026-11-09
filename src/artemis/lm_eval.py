"""Tokenizer-independent held-out language-modelling evaluation.

Reports token-level loss/perplexity and **bits per UTF-8 byte** (BPB). BPB lets
checkpoints with different tokenizers or vocabularies be compared on the same
held-out documents, which raw token loss cannot do.

Each document is scored independently in non-overlapping ``block_size`` windows
(the first token of each window has no in-window context, so BPB is a slight
upper bound). The end-of-document token is scored but contributes no bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import torch
from torch.nn import functional as F

from artemis.config import ModelConfig
from artemis.generate import load_tokenizer
from artemis.model import ArtemisLM


def read_documents(path: Path, *, text_field: str = "text", limit: int | None = None) -> list[str]:
    documents: list[str] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            text = json.loads(line).get(text_field)
            if isinstance(text, str) and text.strip():
                documents.append(text)
            if limit is not None and len(documents) >= limit:
                break
    return documents


@torch.no_grad()
def evaluate_documents(
    model: ArtemisLM,
    tokenizer,
    documents: Iterable[str],
    *,
    block_size: int,
    device: torch.device | str = "cpu",
) -> dict[str, float | int]:
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    total_bytes = 0
    total_documents = 0
    for text in documents:
        ids = tokenizer.encode(text, add_eos=True)
        if len(ids) < 2:
            continue
        total_documents += 1
        total_bytes += len(text.encode("utf-8"))
        for start in range(0, len(ids) - 1, block_size):
            chunk = ids[start:start + block_size + 1]
            if len(chunk) < 2:
                continue
            inputs = torch.tensor([chunk[:-1]], dtype=torch.long, device=device)
            targets = torch.tensor([chunk[1:]], dtype=torch.long, device=device)
            logits, _ = model(inputs)
            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1), reduction="sum"
            )
            total_nll += float(nll)
            total_tokens += targets.numel()
    if total_tokens == 0 or total_bytes == 0:
        raise ValueError("no evaluable documents")
    loss = total_nll / total_tokens
    return {
        "documents": total_documents,
        "tokens": total_tokens,
        "bytes": total_bytes,
        "token_loss": loss,
        "perplexity": math.exp(min(loss, 50.0)),
        "bits_per_byte": total_nll / (math.log(2) * total_bytes),
        "bytes_per_token": total_bytes / total_tokens,
    }


def load_checkpoint(path: Path, device: str = "cpu"):
    payload = torch.load(path, map_location=device, weights_only=True)
    tokenizer = load_tokenizer(payload)
    config = ModelConfig.from_dict(payload["model_config"])
    model = ArtemisLM(config).to(device)
    model.load_state_dict(payload["model_state_dict"])
    return payload, tokenizer, model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, nargs="+", required=True, help="held-out JSONL files")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--max-documents", type=int, default=500)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload, tokenizer, model = load_checkpoint(args.checkpoint, args.device)
    report = {
        "checkpoint": args.checkpoint.name,
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "checkpoint_step": payload.get("step"),
        "parameters": sum(p.numel() for p in model.parameters()),
        "datasets": {},
    }
    for path in args.data:
        documents = read_documents(path, text_field=args.text_field, limit=args.max_documents)
        result = evaluate_documents(
            model, tokenizer, documents, block_size=model.config.block_size, device=args.device
        )
        result["data_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        report["datasets"][str(path)] = result
    text = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

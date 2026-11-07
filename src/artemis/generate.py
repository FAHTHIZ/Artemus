"""Generate text from an Artemis training checkpoint."""
from __future__ import annotations

import argparse
import base64
from pathlib import Path

import torch

from artemis.config import ModelConfig
from artemis.model import ArtemisLM
from artemis.tokenizer import ByteTokenizer, SentencePieceTokenizer
from artemis.train import choose_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--no-cache", action="store_true", help="disable caching for full-context reference comparisons"
    )
    parser.add_argument(
        "--dynamic-cache", action="store_true",
        help="use dynamically concatenated K/V tensors instead of preallocated static buffers",
    )
    return parser.parse_args()


def load_tokenizer(payload: dict) -> ByteTokenizer | SentencePieceTokenizer:
    tokenizer_name = payload.get("tokenizer")
    if tokenizer_name == ByteTokenizer.name:
        return ByteTokenizer()
    if tokenizer_name == SentencePieceTokenizer.name:
        encoded = payload.get("tokenizer_model_b64")
        if not isinstance(encoded, str):
            raise SystemExit("Checkpoint is missing the SentencePiece tokenizer model")
        try:
            model_proto = base64.b64decode(encoded, validate=True)
        except ValueError as exc:
            raise SystemExit("Checkpoint contains an invalid tokenizer model") from exc
        return SentencePieceTokenizer.from_serialized_model(model_proto)
    raise SystemExit(f"Unsupported tokenizer in checkpoint: {tokenizer_name!r}")


def main() -> None:
    args = parse_args()
    if args.tokens < 0:
        raise SystemExit("--tokens must be non-negative")
    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint does not exist: {args.checkpoint}")

    device = choose_device(args.device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if payload.get("format_version") not in {1, 2, 3}:
        raise SystemExit("Unsupported checkpoint format")

    tokenizer = load_tokenizer(payload)
    config = ModelConfig.from_dict(payload["model_config"])
    if config.vocab_size != tokenizer.vocab_size:
        raise SystemExit("Checkpoint model and tokenizer vocabulary sizes do not match")

    model = ArtemisLM(config).to(device)
    model.load_state_dict(payload["model_state_dict"])

    prompt_ids = tokenizer.encode(args.prompt)
    if not prompt_ids:
        raise SystemExit("Prompt must contain at least one token")
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    output_ids = model.generate(
        input_ids,
        args.tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        eos_token_id=tokenizer.stop_token_id,
        use_cache=not args.no_cache,
        use_static_cache=not args.dynamic_cache,
    )[0].tolist()
    print(tokenizer.decode(output_ids))
    print("\nNOTE: generated text is experimental; evaluate it before treating it as useful advice.")


if __name__ == "__main__":
    main()

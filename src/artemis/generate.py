"""Generate a sample from a saved Artemis smoke-test checkpoint."""

import argparse
from pathlib import Path

import torch

from artemis.config import ModelConfig
from artemis.model import ArtemisLM
from artemis.tokenizer import ByteTokenizer
from artemis.train import choose_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.tokens < 0:
        raise SystemExit("--tokens must be non-negative")
    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint does not exist: {args.checkpoint}")

    device = choose_device(args.device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if payload.get("format_version") != 1 or payload.get("tokenizer") != "utf8-byte-v1":
        raise SystemExit("Unsupported checkpoint or tokenizer format")

    config = ModelConfig.from_dict(payload["model_config"])
    model = ArtemisLM(config).to(device)
    model.load_state_dict(payload["model_state_dict"])

    tokenizer = ByteTokenizer()
    prompt_ids = tokenizer.encode(args.prompt)
    if not prompt_ids:
        raise SystemExit("Prompt must contain at least one UTF-8 byte")
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    output_ids = model.generate(
        input_ids,
        args.tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        eos_token_id=tokenizer.eos_token_id,
    )[0].tolist()
    print(tokenizer.decode(output_ids))
    print("\nNOTE: output is from a tiny smoke-test model and should not be treated as reliable advice.")


if __name__ == "__main__":
    main()

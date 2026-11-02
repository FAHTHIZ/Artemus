"""Train an Artemis decoder-only language model on a UTF-8 text corpus."""
from __future__ import annotations

import argparse
import base64
import random
from pathlib import Path

import torch
from torch import nn

from artemis.config import ModelConfig
from artemis.model import ArtemisLM
from artemis.tokenizer import ByteTokenizer, SentencePieceTokenizer


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_tokens(
    path: Path,
    validation_path: Path | None,
    tokenizer: ByteTokenizer | SentencePieceTokenizer,
) -> tuple[torch.Tensor, torch.Tensor]:
    def read_ids(source: Path) -> torch.Tensor:
        if not source.is_file():
            raise FileNotFoundError(f"Dataset text file does not exist: {source}")
        text = source.read_text(encoding="utf-8")
        ids = torch.tensor(tokenizer.encode(text, add_eos=True), dtype=torch.long)
        if ids.numel() < 40:
            raise ValueError(f"Dataset split is too short (at least 40 tokens): {source}")
        return ids

    train_ids = read_ids(path)
    if validation_path is not None:
        return train_ids, read_ids(validation_path)

    split_at = int(train_ids.numel() * 0.9)
    return train_ids[:split_at], train_ids[split_at:]


def build_response_loss_mask(
    token_ids: torch.Tensor, tokenizer: SentencePieceTokenizer
) -> torch.Tensor:
    """Mark assistant text and end-of-turn tokens; ignore user prompts and role markers."""
    if token_ids.ndim != 1:
        raise ValueError("token_ids must be a one-dimensional token stream")

    processor = tokenizer.processor
    prompter_id = int(processor.piece_to_id("<|prompter|>"))
    assistant_id = int(processor.piece_to_id("<|assistant|>"))
    end_id = int(processor.piece_to_id("<|end|>"))
    if min(prompter_id, assistant_id, end_id) < 0:
        raise ValueError("tokenizer is missing required chat control tokens")

    control_tokens = (
        (token_ids == prompter_id)
        | (token_ids == assistant_id)
        | (token_ids == end_id)
    )
    positions = torch.arange(token_ids.numel(), device=token_ids.device)
    last_control_positions = torch.cummax(
        torch.where(control_tokens, positions, torch.full_like(positions, -1)),
        dim=0,
    ).values
    previous_control_positions = torch.cat(
        (
            torch.full((1,), -1, dtype=torch.long, device=token_ids.device),
            last_control_positions[:-1],
        )
    )
    last_control_ids = token_ids[last_control_positions.clamp_min(0)]
    previous_control_ids = token_ids[previous_control_positions.clamp_min(0)]

    # Text after an assistant marker is supervised. The end marker itself is
    # supervised using the immediately preceding control marker's state.
    in_assistant_region = (
        (last_control_positions >= 0) & (last_control_ids == assistant_id)
    )
    end_of_assistant_turn = (
        (token_ids == end_id) & (previous_control_ids == assistant_id)
    )
    return (in_assistant_region | end_of_assistant_turn) & ~(
        (token_ids == prompter_id) | (token_ids == assistant_id)
    )


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: torch.device,
    target_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = data.numel() - block_size - 1
    if max_start < 1:
        raise ValueError(
            f"dataset split has {data.numel()} tokens; needs more than "
            f"block_size + 1 ({block_size + 1})"
        )
    if target_mask is not None and (
        target_mask.ndim != 1 or target_mask.numel() != data.numel()
    ):
        raise ValueError("target_mask must be one-dimensional and match data length")

    starts = torch.randint(0, max_start + 1, (batch_size,))
    offsets = torch.arange(1, block_size + 1)
    if target_mask is not None:
        valid_targets = target_mask[starts[:, None] + offsets[None, :]]
        invalid_rows = ~valid_targets.any(dim=1)
        for _ in range(32):
            if not invalid_rows.any():
                break
            starts[invalid_rows] = torch.randint(
                0, max_start + 1, (int(invalid_rows.sum().item()),)
            )
            valid_targets = target_mask[starts[:, None] + offsets[None, :]]
            invalid_rows = ~valid_targets.any(dim=1)
        if invalid_rows.any():
            raise RuntimeError(
                "Could not sample a block containing assistant response targets"
            )

    x = torch.stack([data[i : i + block_size] for i in starts.tolist()])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts.tolist()])
    if target_mask is not None:
        y = y.masked_fill(~valid_targets, -100)
    return x.to(device), y.to(device)


@torch.inference_mode()
def estimate_loss(
    model: ArtemisLM,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: torch.device,
    eval_iters: int,
    train_mask: torch.Tensor | None = None,
    val_mask: torch.Tensor | None = None,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    results: dict[str, float] = {}
    try:
        for name, split, mask in (
            ("train", train_data, train_mask),
            ("val", val_data, val_mask),
        ):
            losses = []
            for _ in range(eval_iters):
                x, y = get_batch(split, batch_size, block_size, device, mask)
                _, loss = model(x, y)
                if loss is None or not torch.isfinite(loss):
                    raise RuntimeError("loss is missing or non-finite during evaluation")
                losses.append(float(loss.item()))
            results[name] = sum(losses) / len(losses)
    finally:
        model.train(was_training)
    return results


def tokenizer_checkpoint_fields(
    tokenizer: ByteTokenizer | SentencePieceTokenizer,
) -> dict[str, str | None]:
    if isinstance(tokenizer, SentencePieceTokenizer):
        serialized = base64.b64encode(tokenizer.serialized_model).decode("ascii")
        return {"tokenizer": tokenizer.name, "tokenizer_model_b64": serialized}
    return {"tokenizer": tokenizer.name, "tokenizer_model_b64": None}


def save_checkpoint(
    path: Path,
    model: ArtemisLM,
    optimizer: torch.optim.Optimizer,
    tokenizer: ByteTokenizer | SentencePieceTokenizer,
    step: int,
    losses: dict[str, float],
    *,
    response_only: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 2,
            "step": step,
            "model_config": model.config.to_dict(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "losses": losses,
            "response_only": response_only,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            **tokenizer_checkpoint_fields(tokenizer),
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="UTF-8 training text file")
    parser.add_argument("--validation-data", type=Path, help="separate validation text file")
    parser.add_argument("--output", type=Path, default=Path("artifacts/artemis-tiny.pt"))
    parser.add_argument("--resume", type=Path, help="resume from a checkpoint created by this trainer")
    parser.add_argument("--tokenizer", choices=("byte", "sentencepiece"), default="byte")
    parser.add_argument("--tokenizer-model", type=Path, help="existing SentencePiece .model file")
    parser.add_argument("--vocab-size", type=int, default=2048)
    parser.add_argument(
        "--response-only",
        action="store_true",
        help="compute next-token loss only over assistant replies and end-of-turn tokens",
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--embedding-size", type=int, default=64)
    parser.add_argument("--kv-heads", type=int, help="GQA key/value head count; defaults to full MHA")
    parser.add_argument("--use-rope", action="store_true", help="use rotary position embeddings")
    parser.add_argument("--use-qk-norm", action="store_true", help="normalise query/key heads")
    parser.add_argument("--use-swiglu", action="store_true", help="use a gated SwiGLU feed-forward block")
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--eval-iters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, mps, or a torch device")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps < 1 or args.eval_interval < 1 or args.eval_iters < 1:
        raise SystemExit("steps, eval-interval, and eval-iters must be positive")
    if args.batch_size < 1 or args.learning_rate <= 0:
        raise SystemExit("batch-size must be positive and learning-rate must be > 0")
    if args.block_size < 2:
        raise SystemExit("block-size must be at least 2")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = choose_device(args.device)
    resume_payload = None
    if args.resume is not None:
        if not args.resume.is_file():
            raise SystemExit(f"Checkpoint does not exist: {args.resume}")
        resume_payload = torch.load(args.resume, map_location=device, weights_only=True)
        if resume_payload.get("format_version") not in {1, 2}:
            raise SystemExit("Unsupported checkpoint format")

    if args.tokenizer == "sentencepiece":
        if resume_payload is not None:
            if resume_payload.get("tokenizer") != SentencePieceTokenizer.name:
                raise SystemExit("Resume checkpoint does not contain a SentencePiece tokenizer")
            encoded = resume_payload.get("tokenizer_model_b64")
            if not isinstance(encoded, str):
                raise SystemExit("Resume checkpoint is missing its SentencePiece model")
            tokenizer = SentencePieceTokenizer.from_serialized_model(
                base64.b64decode(encoded, validate=True)
            )
        elif args.tokenizer_model is not None:
            tokenizer = SentencePieceTokenizer(model_file=args.tokenizer_model)
        else:
            prefix = args.output.parent / f"{args.output.stem}-tokenizer"
            tokenizer = SentencePieceTokenizer.train(
                args.data, prefix, vocab_size=args.vocab_size
            )
    else:
        if args.tokenizer_model is not None:
            raise SystemExit("--tokenizer-model can only be used with --tokenizer sentencepiece")
        if resume_payload is not None and resume_payload.get("tokenizer") != ByteTokenizer.name:
            raise SystemExit("Resume checkpoint uses a different tokenizer")
        tokenizer = ByteTokenizer()

    if args.response_only and not isinstance(tokenizer, SentencePieceTokenizer):
        raise SystemExit("--response-only currently requires --tokenizer sentencepiece")

    if resume_payload is not None:
        if bool(resume_payload.get("response_only", False)) != args.response_only:
            raise SystemExit("Resume checkpoint response_only setting does not match this run")
        config = ModelConfig.from_dict(resume_payload["model_config"])
    else:
        config = ModelConfig(
            vocab_size=tokenizer.vocab_size,
            block_size=args.block_size,
            n_layer=args.layers,
            n_head=args.heads,
            n_embd=args.embedding_size,
            dropout=0.0,
            n_kv_head=args.kv_heads,
            use_rope=args.use_rope,
            use_qk_norm=args.use_qk_norm,
            use_swiglu=args.use_swiglu,
            rope_theta=args.rope_theta,
        )
    if config.vocab_size != tokenizer.vocab_size:
        raise SystemExit(
            f"Tokenizer vocabulary ({tokenizer.vocab_size}) does not match model "
            f"configuration ({config.vocab_size})"
        )

    train_data, val_data = load_tokens(args.data, args.validation_data, tokenizer)
    if train_data.numel() <= config.block_size + 1:
        raise SystemExit("Training split is too small for block-size; reduce block-size or add data")
    if val_data.numel() <= config.block_size + 1:
        raise SystemExit("Validation split is too small for block-size; reduce block-size or add data")

    train_mask = val_mask = None
    if args.response_only:
        train_mask = build_response_loss_mask(train_data, tokenizer)
        val_mask = build_response_loss_mask(val_data, tokenizer)
        if not train_mask.any() or not val_mask.any():
            raise SystemExit("Response-only masking found no assistant targets in a data split")
        print(
            f"response_only=true train_target_tokens={int(train_mask.sum())} "
            f"validation_target_tokens={int(val_mask.sum())}"
        )

    model = ArtemisLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    start_step = 0
    if resume_payload is not None:
        model.load_state_dict(resume_payload["model_state_dict"])
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        start_step = int(resume_payload["step"])
        rng_state = resume_payload.get("torch_rng_state")
        if rng_state is not None:
            torch.set_rng_state(rng_state.cpu())
        cuda_rng_states = resume_payload.get("cuda_rng_states", [])
        if device.type == "cuda" and cuda_rng_states:
            torch.cuda.set_rng_state_all(cuda_rng_states)
        if args.steps <= start_step:
            raise SystemExit(
                f"--steps ({args.steps}) must be greater than checkpoint step ({start_step})"
            )

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"device={device} parameters={parameter_count:,} tokenizer={tokenizer.name} "
        f"vocab_size={tokenizer.vocab_size} train_tokens={train_data.numel():,} "
        f"validation_tokens={val_data.numel():,} start_step={start_step}"
    )

    latest_losses: dict[str, float] = {}
    for step in range(start_step + 1, args.steps + 1):
        model.train()
        x, y = get_batch(
            train_data, args.batch_size, config.block_size, device, train_mask
        )
        _, loss = model(x, y)
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError(f"non-finite or missing training loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step == 1 or step % args.eval_interval == 0 or step == args.steps:
            latest_losses = estimate_loss(
                model, train_data, val_data, args.batch_size,
                config.block_size, device, args.eval_iters, train_mask, val_mask,
            )
            print(
                f"step={step:>5} train_loss={latest_losses['train']:.4f} "
                f"val_loss={latest_losses['val']:.4f}"
            )
            save_checkpoint(
                args.output, model, optimizer, tokenizer, step, latest_losses,
                response_only=args.response_only,
            )

    print(f"checkpoint_saved={args.output.resolve()}")
    print("NOTE: capability must be established by held-out task evaluations, not loss alone.")


if __name__ == "__main__":
    main()

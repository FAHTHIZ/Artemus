"""Train an Artemis decoder-only language model on a UTF-8 text corpus."""
from __future__ import annotations

import argparse
import base64
import hashlib
from contextlib import nullcontext
import os
import random
from datetime import datetime, timezone
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
    *,
    document_format: str = "text",
    text_field: str = "text",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a corpus; JSONL mode appends EOS after every independently curated document."""
    if document_format not in {"text", "jsonl"}:
        raise ValueError("document_format must be 'text' or 'jsonl'")
    if not text_field.strip():
        raise ValueError("text_field must not be empty")

    def read_ids(source: Path) -> torch.Tensor:
        if not source.is_file():
            raise FileNotFoundError(f"Dataset file does not exist: {source}")
        if document_format == "text":
            text = source.read_text(encoding="utf-8")
            ids_list = tokenizer.encode(text, add_eos=True)
        else:
            import json

            ids_list: list[int] = []
            with source.open("r", encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, start=1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid JSONL at {source}:{line_number}") from exc
                    if not isinstance(row, dict) or not isinstance(row.get(text_field), str):
                        raise ValueError(
                            f"JSONL row at {source}:{line_number} must contain string field "
                            f"{text_field!r}"
                        )
                    document = row[text_field].strip()
                    if document:
                        ids_list.extend(tokenizer.encode(document, add_eos=True))
        ids = torch.tensor(ids_list, dtype=torch.long)
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_resume_data(
    *,
    args: argparse.Namespace,
    resume_payload: dict | None,
) -> bool:
    """Require exact dataset/manifest identity when continuing a checkpoint."""
    if resume_payload is None:
        return False
    metadata = resume_payload.get("run_metadata") or {}
    runs = metadata.get("runs") or []
    if not runs:
        raise SystemExit(
            "Resume checkpoint has no auditable run metadata; refusing an unverified continuation"
        )
    parent = runs[-1]
    current = {
        "train_text_sha256": sha256_file(args.data),
        "validation_text_sha256": (
            sha256_file(args.validation_data) if args.validation_data is not None else None
        ),
        "data_manifest_sha256": (
            sha256_file(args.data_manifest)
            if getattr(args, "data_manifest", None) is not None
            else None
        ),
    }
    differences = {
        key: {"checkpoint": parent.get(key), "current": value}
        for key, value in current.items()
        if parent.get(key) != value
    }
    if differences and not args.allow_data_change:
        raise SystemExit(
            "Resume data/manifest hashes differ from the checkpoint. "
            "Refusing to silently change the training distribution. "
            "Pass --allow-data-change only for a deliberate curriculum transition. "
            + str(differences)
        )
    return bool(differences)


def learning_rate_at_step(
    *,
    step: int,
    start_step: int,
    target_step: int,
    base_learning_rate: float,
    min_learning_rate: float,
    warmup_steps: int,
    schedule: str,
) -> float:
    if schedule == "constant":
        return base_learning_rate
    if schedule != "cosine":
        raise ValueError(f"unsupported learning-rate schedule: {schedule}")
    if not 0.0 <= min_learning_rate <= base_learning_rate:
        raise ValueError("min_learning_rate must be between zero and the base learning rate")
    elapsed = max(0, step - start_step)
    if warmup_steps > 0 and elapsed <= warmup_steps:
        fraction = elapsed / warmup_steps
        return min_learning_rate + fraction * (base_learning_rate - min_learning_rate)
    decay_steps = max(1, target_step - start_step - warmup_steps)
    decay_elapsed = min(decay_steps, max(0, elapsed - warmup_steps))
    progress = decay_elapsed / decay_steps
    import math
    return min_learning_rate + 0.5 * (base_learning_rate - min_learning_rate) * (
        1.0 + math.cos(math.pi * progress)
    )


def build_checkpoint_metadata(
    *,
    args: argparse.Namespace,
    config: ModelConfig,
    tokenizer: ByteTokenizer | SentencePieceTokenizer,
    device: torch.device,
    start_step: int,
    resume_payload: dict | None,
    parent_checkpoint_sha256: str | None,
) -> dict:
    """Build an auditable, weights-only-safe record of model and data lineage."""
    train_hash = sha256_file(args.data)
    validation_hash = (
        sha256_file(args.validation_data) if args.validation_data is not None else None
    )
    manifest_path = getattr(args, "data_manifest", None)
    manifest_hash = (
        sha256_file(manifest_path) if manifest_path is not None else None
    )

    previous_metadata = (
        resume_payload.get("run_metadata", {}) if resume_payload is not None else {}
    )
    previous_lineage = previous_metadata.get("lineage", {})
    if resume_payload is None:
        lineage = {
            "origin": "random_initialization",
            "initialization_seed": int(args.seed),
            "root_source_commit": os.environ.get("GITHUB_SHA"),
            "root_parent_checkpoint_sha256": None,
        }
        previous_runs = []
    elif previous_lineage:
        lineage = dict(previous_lineage)
        previous_runs = list(previous_metadata.get("runs", []))
    else:
        # A legacy checkpoint can be used, but its initialization ancestry is
        # not reconstructed from assumptions or documentation alone.
        lineage = {
            "origin": "legacy_checkpoint_origin_unverified",
            "initialization_seed": None,
            "root_source_commit": None,
            "root_parent_checkpoint_sha256": parent_checkpoint_sha256,
        }
        previous_runs = list(previous_metadata.get("runs", []))

    run_record = {
        "phase": str(getattr(args, "phase", "unspecified")),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": os.environ.get("GITHUB_SHA"),
        "workflow_run_id": os.environ.get("GITHUB_RUN_ID"),
        "workflow_name": os.environ.get("GITHUB_WORKFLOW"),
        "seed": int(args.seed),
        "start_step": int(start_step),
        "target_step": int(args.steps),
        "optimizer": "AdamW",
        "learning_rate": float(args.learning_rate),
        "learning_rate_schedule": str(args.lr_schedule),
        "minimum_learning_rate": float(args.min_learning_rate),
        "warmup_steps": int(args.warmup_steps),
        "data_changed_from_parent": bool(
            getattr(args, "_resume_data_changed", False)
        ),
        "batch_size": int(args.batch_size),
        "block_size": int(config.block_size),
        "device": str(device),
        "response_only": bool(args.response_only),
        "document_format": str(getattr(args, "document_format", "text")),
        "text_field": str(getattr(args, "text_field", "text")),
        "precision": str(getattr(args, "precision", "fp32")),
        "gradient_accumulation_steps": int(
            getattr(args, "gradient_accumulation_steps", 1)
        ),
        "estimated_tokens_per_optimizer_step": int(
            args.batch_size * config.block_size
            * int(getattr(args, "gradient_accumulation_steps", 1))
        ),
        "estimated_tokens_processed_this_run": int(
            max(0, args.steps - start_step)
            * args.batch_size
            * config.block_size
            * int(getattr(args, "gradient_accumulation_steps", 1))
        ),
        "objective_changed_from_parent": bool(
            resume_payload is not None
            and bool(resume_payload.get("response_only", False)) != bool(args.response_only)
        ),
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
        "train_text_sha256": train_hash,
        "validation_text_sha256": validation_hash,
        "data_manifest_sha256": manifest_hash,
        "tokenizer": tokenizer.name,
        "tokenizer_vocab_size": int(tokenizer.vocab_size),
        "tokenizer_model_sha256": (
            hashlib.sha256(tokenizer.serialized_model).hexdigest()
            if isinstance(tokenizer, SentencePieceTokenizer)
            else None
        ),
        "model_config": config.to_dict(),
    }

    return {
        "lineage": lineage,
        "runs": previous_runs + [run_record],
    }


def save_checkpoint(
    path: Path,
    model: ArtemisLM,
    optimizer: torch.optim.Optimizer,
    tokenizer: ByteTokenizer | SentencePieceTokenizer,
    step: int,
    losses: dict[str, float],
    *,
    response_only: bool,
    run_metadata: dict,
    grad_scaler_state_dict: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 3,
            "step": step,
            "model_config": model.config.to_dict(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "losses": losses,
            "response_only": response_only,
            "run_metadata": run_metadata,
            "grad_scaler_state_dict": grad_scaler_state_dict,
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
    parser.add_argument("--data-manifest", type=Path, help="source manifest used to create the training text")
    parser.add_argument(
        "--document-format",
        choices=("text", "jsonl"),
        default="text",
        help="JSONL mode reads one independent document from --text-field and appends EOS per document",
    )
    parser.add_argument("--text-field", default="text", help="string field containing each JSONL document")
    parser.add_argument(
        "--phase",
        choices=("foundation_pretraining", "continued_pretraining", "instruction_sft", "task_sft", "post_training", "unspecified"),
        default="unspecified",
    )
    parser.add_argument(
        "--allow-objective-change",
        action="store_true",
        help="allow switching response-only loss when resuming from a checkpoint",
    )
    parser.add_argument(
        "--allow-data-change",
        action="store_true",
        help="allow a deliberate dataset/manifest change when resuming; record it in lineage",
    )
    parser.add_argument(
        "--override-resume-learning-rate",
        action="store_true",
        help="apply --learning-rate after restoring the optimizer state",
    )
    parser.add_argument(
        "--lr-schedule",
        choices=("constant", "cosine"),
        default="constant",
        help="learning-rate schedule for this run segment",
    )
    parser.add_argument("--min-learning-rate", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=0)
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
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="microbatches per optimiser update; increases effective batch size",
    )
    parser.add_argument(
        "--precision",
        choices=("fp32", "fp16", "bf16"),
        default="fp32",
        help="CUDA mixed precision; use fp16 for T4-class GPUs or bf16 on supported accelerators",
    )
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--embedding-size", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, help="factorised token embedding width; defaults to model width")
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
    if args.gradient_accumulation_steps < 1:
        raise SystemExit("gradient-accumulation-steps must be positive")
    if args.block_size < 2:
        raise SystemExit("block-size must be at least 2")
    if args.warmup_steps < 0 or args.min_learning_rate < 0:
        raise SystemExit("warmup-steps and min-learning-rate must be non-negative")
    if args.lr_schedule == "cosine" and args.min_learning_rate > args.learning_rate:
        raise SystemExit("min-learning-rate must not exceed learning-rate for cosine scheduling")

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
        if resume_payload.get("format_version") not in {1, 2, 3}:
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
        if (
            bool(resume_payload.get("response_only", False)) != args.response_only
            and not args.allow_objective_change
        ):
            raise SystemExit(
                "Resume checkpoint response_only setting differs; pass "
                "--allow-objective-change to record this explicit phase transition"
            )
        config = ModelConfig.from_dict(resume_payload["model_config"])
    else:
        config = ModelConfig(
            vocab_size=tokenizer.vocab_size,
            block_size=args.block_size,
            n_layer=args.layers,
            n_head=args.heads,
            n_embd=args.embedding_size,
            dropout=0.0,
            embedding_dim=args.embedding_dim,
            n_kv_head=args.kv_heads,
            use_rope=args.use_rope,
            use_qk_norm=args.use_qk_norm,
            use_swiglu=args.use_swiglu,
            rope_theta=args.rope_theta,
        )
    resume_data_changed = validate_resume_data(
        args=args,
        resume_payload=resume_payload,
    )
    args._resume_data_changed = resume_data_changed

    if config.vocab_size != tokenizer.vocab_size:
        raise SystemExit(
            f"Tokenizer vocabulary ({tokenizer.vocab_size}) does not match model "
            f"configuration ({config.vocab_size})"
        )

    train_data, val_data = load_tokens(
        args.data,
        args.validation_data,
        tokenizer,
        document_format=args.document_format,
        text_field=args.text_field,
    )
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
    parent_checkpoint_sha256 = (
        sha256_file(args.resume) if args.resume is not None else None
    )
    if resume_payload is not None:
        model.load_state_dict(resume_payload["model_state_dict"])
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        if args.override_resume_learning_rate:
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = args.learning_rate
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
        f"validation_tokens={val_data.numel():,} start_step={start_step} "
        f"precision={args.precision} grad_accum={args.gradient_accumulation_steps}"
    )

    run_metadata = build_checkpoint_metadata(
        args=args,
        config=config,
        tokenizer=tokenizer,
        device=device,
        start_step=start_step,
        resume_payload=resume_payload,
        parent_checkpoint_sha256=parent_checkpoint_sha256,
    )

    if args.precision != "fp32" and device.type != "cuda":
        raise SystemExit("--precision fp16/bf16 currently requires a CUDA device")
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise SystemExit("This CUDA GPU does not report bfloat16 support; use --precision fp16")
    use_amp = args.precision != "fp32"
    amp_dtype = torch.float16 if args.precision == "fp16" else torch.bfloat16
    scaler = (
        torch.cuda.amp.GradScaler(enabled=True)
        if args.precision == "fp16"
        else None
    )
    if scaler is not None and resume_payload is not None:
        scaler_state = resume_payload.get("grad_scaler_state_dict")
        if isinstance(scaler_state, dict):
            scaler.load_state_dict(scaler_state)

    latest_losses: dict[str, float] = {}
    for step in range(start_step + 1, args.steps + 1):
        scheduled_lr = learning_rate_at_step(
            step=step,
            start_step=start_step,
            target_step=args.steps,
            base_learning_rate=args.learning_rate,
            min_learning_rate=args.min_learning_rate,
            warmup_steps=args.warmup_steps,
            schedule=args.lr_schedule,
        )
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = scheduled_lr
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for micro_step in range(args.gradient_accumulation_steps):
            x, y = get_batch(
                train_data, args.batch_size, config.block_size, device, train_mask
            )
            with (
                torch.autocast(device_type="cuda", dtype=amp_dtype)
                if use_amp
                else nullcontext()
            ):
                _, loss = model(x, y)
                if loss is None or not torch.isfinite(loss):
                    raise RuntimeError(
                        f"non-finite or missing training loss at step {step}, "
                        f"micro-step {micro_step + 1}"
                    )
                scaled_loss = loss / args.gradient_accumulation_steps
            if scaler is not None:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

        if scaler is not None:
            scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        if step == 1 or step % args.eval_interval == 0 or step == args.steps:
            latest_losses = estimate_loss(
                model, train_data, val_data, args.batch_size,
                config.block_size, device, args.eval_iters, train_mask, val_mask,
            )
            print(
                f"step={step:>5} train_loss={latest_losses['train']:.4f} "
                f"val_loss={latest_losses['val']:.4f} lr={scheduled_lr:.7f}"
            )
            save_checkpoint(
                args.output, model, optimizer, tokenizer, step, latest_losses,
                response_only=args.response_only,
                run_metadata=run_metadata,
                grad_scaler_state_dict=scaler.state_dict() if scaler is not None else None,
            )

    print(f"checkpoint_saved={args.output.resolve()}")
    print("NOTE: capability must be established by held-out task evaluations, not loss alone.")


if __name__ == "__main__":
    main()

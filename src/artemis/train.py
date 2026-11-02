"""Train a tiny Artemis baseline on a local UTF-8 text file."""

import argparse
import random
from pathlib import Path

import torch
from torch import nn

from artemis.config import ModelConfig
from artemis.model import ArtemisLM
from artemis.tokenizer import ByteTokenizer


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_tokens(
    path: Path, validation_path: Path | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    def read_ids(source: Path) -> torch.Tensor:
        if not source.is_file():
            raise FileNotFoundError(f"Dataset text file does not exist: {source}")
        text = source.read_text(encoding="utf-8")
        ids = torch.tensor(ByteTokenizer().encode(text, add_eos=True), dtype=torch.long)
        if ids.numel() < 40:
            raise ValueError(f"Dataset split is too short (at least 40 UTF-8 bytes): {source}")
        return ids

    train_ids = read_ids(path)
    if validation_path is not None:
        # Prefer group-separated, externally prepared splits when provided.
        return train_ids, read_ids(validation_path)

    # Backward-compatible fallback for small local smoke-test corpora.
    split_at = int(train_ids.numel() * 0.9)
    return train_ids[:split_at], train_ids[split_at:]


def get_batch(
    data: torch.Tensor, batch_size: int, block_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = data.numel() - block_size - 1
    if max_start < 1:
        raise ValueError(
            f"dataset split has {data.numel()} tokens; needs more than "
            f"block_size + 1 ({block_size + 1})"
        )
    starts = torch.randint(0, max_start + 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in starts.tolist()])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts.tolist()])
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
) -> dict[str, float]:
    model.eval()
    results: dict[str, float] = {}
    for name, split in (("train", train_data), ("val", val_data)):
        losses = []
        for _ in range(eval_iters):
            x, y = get_batch(split, batch_size, block_size, device)
            _, loss = model(x, y)
            if loss is None:
                raise RuntimeError("loss unexpectedly missing during evaluation")
            losses.append(float(loss.item()))
        results[name] = sum(losses) / len(losses)
    model.train()
    return results


def save_checkpoint(
    path: Path,
    model: ArtemisLM,
    optimizer: torch.optim.Optimizer,
    step: int,
    losses: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "step": step,
            "model_config": model.config.to_dict(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "losses": losses,
            "tokenizer": "utf8-byte-v1",
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="UTF-8 plain-text training file")
    parser.add_argument("--validation-data", type=Path, help="separate validation file (preferred)")
    parser.add_argument("--output", type=Path, default=Path("artifacts/artemis-tiny.pt"))
    parser.add_argument("--resume", type=Path, help="resume from a checkpoint created by this trainer")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--embedding-size", type=int, default=64)
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
        if resume_payload.get("format_version") != 1:
            raise SystemExit("Unsupported checkpoint format")
        config = ModelConfig.from_dict(resume_payload["model_config"])
    else:
        config = ModelConfig(
            block_size=args.block_size,
            n_layer=args.layers,
            n_head=args.heads,
            n_embd=args.embedding_size,
            dropout=0.0,
        )
    train_data, val_data = load_tokens(args.data, args.validation_data)
    if train_data.numel() <= config.block_size + 1:
        raise SystemExit("Training split is too small for block-size; reduce block-size or add data")
    if val_data.numel() <= config.block_size + 1:
        raise SystemExit("Validation split is too small for block-size; reduce block-size or add data")

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
    print(f"device={device} parameters={parameter_count:,} train_tokens={train_data.numel():,} "
          f"validation_tokens={val_data.numel():,} start_step={start_step}")

    latest_losses: dict[str, float] = {}
    for step in range(start_step + 1, args.steps + 1):
        model.train()
        x, y = get_batch(train_data, args.batch_size, config.block_size, device)
        _, loss = model(x, y)
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError(f"non-finite or missing training loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step == 1 or step % args.eval_interval == 0 or step == args.steps:
            latest_losses = estimate_loss(
                model,
                train_data,
                val_data,
                args.batch_size,
                config.block_size,
                device,
                args.eval_iters,
            )
            print(
                f"step={step:>5} train_loss={latest_losses['train']:.4f} "
                f"val_loss={latest_losses['val']:.4f}"
            )
            save_checkpoint(args.output, model, optimizer, step, latest_losses)

    print(f"checkpoint_saved={args.output.resolve()}")
    print("NOTE: this is a pipeline smoke-test model, not a capable assistant.")


if __name__ == "__main__":
    main()

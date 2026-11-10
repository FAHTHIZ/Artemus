"""Crash-safe, auto-resuming supervisor for a configured CPU training stage.

Run from the repository root:
    python scripts/cpu/stage_supervisor.py s3

The stage configuration is resolved beside this file, while data/checkpoint
paths in that configuration are resolved from the repository working directory.
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format_version") != 3:
        raise ValueError(f"Expected a provenance-enabled format-v3 checkpoint: {path}")
    if not isinstance(payload.get("step"), int):
        raise ValueError(f"Checkpoint has no valid global step: {path}")
    if not isinstance(payload.get("model_state_dict"), dict):
        raise ValueError(f"Checkpoint is missing model weights: {path}")
    if not isinstance(payload.get("optimizer_state_dict"), dict):
        raise ValueError(f"Checkpoint is not resumable (optimizer state missing): {path}")
    return payload


def learning_rate_at(
    step: int,
    *,
    start: int,
    target: int,
    peak: float,
    floor: float,
    warmup: int,
) -> float:
    elapsed = step - start
    if elapsed <= warmup:
        return floor + elapsed / max(1, warmup) * (peak - floor)
    decay_steps = max(1, target - start - warmup)
    progress = min(1.0, max(0.0, (elapsed - warmup) / decay_steps))
    return floor + 0.5 * (peak - floor) * (1.0 + math.cos(math.pi * progress))


def main() -> int:
    if len(sys.argv) != 2 or not sys.argv[1].strip():
        raise SystemExit("Usage: stage_supervisor.py STAGE_NAME")

    stage_name = sys.argv[1].strip()
    config_path = Path(__file__).resolve().parent / f"{stage_name}.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Stage configuration does not exist: {config_path}")
    cfg = json.loads(config_path.read_text(encoding="utf-8"))

    repository_root = Path.cwd()
    parent_path = Path(cfg["parent"])
    output_path = Path(cfg["output"])
    if not parent_path.is_absolute():
        parent_path = repository_root / parent_path
    if not output_path.is_absolute():
        output_path = repository_root / output_path

    parent_payload = load_checkpoint(parent_path)
    parent_sha256 = sha256_file(parent_path)
    expected_parent_sha = cfg.get("parent_sha256")
    if expected_parent_sha and parent_sha256 != expected_parent_sha:
        raise ValueError(
            "Parent checkpoint SHA-256 mismatch: refusing to train from an unexpected model"
        )
    expected_parent_step = cfg.get("expected_parent_step")
    if expected_parent_step is not None and parent_payload["step"] != expected_parent_step:
        raise ValueError(
            f"Parent global step is {parent_payload['step']}; expected {expected_parent_step}"
        )
    expected_parent_response_only = cfg.get("expected_parent_response_only")
    if (
        expected_parent_response_only is not None
        and parent_payload.get("response_only") is not expected_parent_response_only
    ):
        raise ValueError("Parent checkpoint has the wrong training objective for this stage")

    start_step = int(parent_payload["step"])
    target_step = start_step + int(cfg["steps"])
    peak_lr = float(cfg["lr"])
    minimum_lr = float(cfg["min_lr"])
    warmup_steps = int(cfg["warmup"])
    max_attempts = int(cfg.get("max_attempts", 50))
    if target_step <= start_step or max_attempts < 1:
        raise ValueError("Stage step budget and retry count must be positive")
    if not 0.0 <= minimum_lr <= peak_lr:
        raise ValueError("Stage minimum learning rate must be between zero and peak learning rate")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not output_path.exists():
        shutil.copy2(parent_path, output_path)

    for attempt in range(max_attempts):
        current_payload = load_checkpoint(output_path)
        current_step = int(current_payload["step"])
        if current_step < start_step or current_step > target_step:
            raise ValueError(
                f"Output checkpoint step {current_step} is outside "
                f"the configured stage range {start_step}..{target_step}"
            )
        if current_step >= target_step:
            break

        command = [
            "artemus-train",
            *cfg["args"],
            "--steps", str(target_step),
            "--resume", str(output_path),
            "--output", str(output_path),
            "--allow-data-change",
            "--override-resume-learning-rate",
            "--lr-schedule", "cosine",
            "--min-learning-rate", str(minimum_lr),
            "--device", "cpu",
        ]
        if current_step == start_step:
            command += [
                "--learning-rate", str(peak_lr),
                "--warmup-steps", str(warmup_steps),
            ]
        else:
            command += [
                "--learning-rate",
                f"{learning_rate_at(current_step, start=start_step, target=target_step, peak=peak_lr, floor=minimum_lr, warmup=warmup_steps):.8g}",
                "--warmup-steps", "0",
            ]

        print(
            f"Stage {stage_name}: attempt {attempt + 1}/{max_attempts}, "
            f"global step {current_step} -> {target_step}",
            flush=True,
        )
        result = subprocess.run(command, check=False)
        print(f"Training process exit code: {result.returncode}", flush=True)

        after_payload = load_checkpoint(output_path)
        after_step = int(after_payload["step"])
        if after_step < current_step or after_step > target_step:
            raise RuntimeError("Trainer wrote an invalid checkpoint step")
        if after_step >= target_step:
            break
        if result.returncode == 0:
            print(
                "Trainer exited successfully before the requested target; "
                "supervisor will resume from the saved step.",
                flush=True,
            )
        else:
            print("Training interrupted; retrying from the latest valid checkpoint.", flush=True)
        time.sleep(5)

    final_payload = load_checkpoint(output_path)
    final_step = int(final_payload["step"])
    if final_step != target_step:
        raise RuntimeError(
            f"Stage {stage_name} did not reach its target: {final_step}/{target_step}"
        )

    expected_response_only = cfg.get("expected_response_only")
    if (
        expected_response_only is not None
        and final_payload.get("response_only") is not expected_response_only
    ):
        raise RuntimeError("Final checkpoint does not have the expected training objective")

    metadata = final_payload.get("run_metadata") or {}
    runs = metadata.get("runs") or []
    if not runs or runs[-1].get("phase") != cfg.get("expected_phase", runs[-1].get("phase")):
        raise RuntimeError("Final checkpoint is missing the expected stage lineage")

    print(
        f"STAGE COMPLETE: {stage_name}; global step {final_step}; "
        f"checkpoint {output_path}; sha256 {sha256_file(output_path)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from argparse import Namespace
from pathlib import Path

import torch
import pytest

from artemis.config import ModelConfig
from artemis.model import ArtemisLM
from artemis.tokenizer import ByteTokenizer
import artemis.train as train_module
from artemis.train import (
    build_checkpoint_metadata,
    checkpoint_loss_history,
    checkpoint_run_metadata,
    estimate_loss,
    get_batch,
    periodic_step_due,
    save_checkpoint,
)


def test_checkpoint_loss_history_preserves_explicit_measurement_step() -> None:
    losses, measured_at = checkpoint_loss_history({
        "step": 200,
        "losses": {"train": 1.2, "val": 1.4},
        "losses_at_step": 150,
    })
    assert losses == {"train": 1.2, "val": 1.4}
    assert measured_at == 150


def test_checkpoint_loss_history_supports_older_checkpoint_formats() -> None:
    losses, measured_at = checkpoint_loss_history({
        "step": 100,
        "losses": {"train": 1.1, "val": 1.3},
    })
    assert losses == {"train": 1.1, "val": 1.3}
    assert measured_at == 100
    assert checkpoint_loss_history(None) == ({}, None)


def test_checkpoint_metadata_records_actual_progress_without_mutating_run() -> None:
    original = {
        "lineage": {"origin": "test"},
        "runs": [{
            "start_step": 100,
            "target_step": 500,
            "estimated_tokens_per_optimizer_step": 4096,
            "estimated_tokens_budget_this_run": 1_638_400,
            "optimizer_steps_completed": 0,
            "estimated_tokens_processed_this_run": 0,
            "checkpoint_step": 100,
            "target_reached": False,
        }],
    }

    partial = checkpoint_run_metadata(original, step=200)
    partial_run = partial["runs"][-1]
    assert partial_run["checkpoint_step"] == 200
    assert partial_run["optimizer_steps_completed"] == 100
    assert partial_run["estimated_tokens_processed_this_run"] == 409_600
    assert partial_run["estimated_tokens_budget_this_run"] == 1_638_400
    assert partial_run["target_reached"] is False
    assert original["runs"][-1]["checkpoint_step"] == 100

    complete = checkpoint_run_metadata(original, step=500)["runs"][-1]
    assert complete["optimizer_steps_completed"] == 400
    assert complete["estimated_tokens_processed_this_run"] == 1_638_400
    assert complete["target_reached"] is True


def test_checkpoint_metadata_rejects_steps_outside_current_run() -> None:
    metadata = {
        "runs": [{"start_step": 10, "target_step": 20,
                  "estimated_tokens_per_optimizer_step": 100}],
    }
    with pytest.raises(ValueError, match="within the current run"):
        checkpoint_run_metadata(metadata, step=9)
    with pytest.raises(ValueError, match="within the current run"):
        checkpoint_run_metadata(metadata, step=21)


def test_evaluation_and_checkpoint_schedules_are_independent() -> None:
    assert periodic_step_due(50, target_step=260, interval=25) is True
    assert periodic_step_due(50, target_step=260, interval=100) is False
    assert periodic_step_due(100, target_step=260, interval=100) is True
    assert periodic_step_due(250, target_step=260, interval=250) is True
    assert periodic_step_due(260, target_step=260, interval=100) is True


def test_periodic_step_due_rejects_invalid_ranges() -> None:
    with pytest.raises(ValueError, match="interval must be positive"):
        periodic_step_due(1, target_step=10, interval=0)
    with pytest.raises(ValueError, match="between 1 and target_step"):
        periodic_step_due(11, target_step=10, interval=5)


def test_get_batch_returns_shifted_sequences() -> None:
    data = torch.arange(100, dtype=torch.long)
    x, y = get_batch(data, batch_size=4, block_size=8, device=torch.device("cpu"))
    assert x.shape == (4, 8)
    assert y.shape == (4, 8)
    assert torch.equal(y[:, :-1], x[:, 1:])


def test_estimate_loss_is_repeatable_and_does_not_consume_training_rng() -> None:
    torch.manual_seed(41)
    config = ModelConfig(vocab_size=257, block_size=16, n_layer=1, n_head=2, n_embd=16)
    model = ArtemisLM(config)
    train_data = torch.arange(256, dtype=torch.long).remainder(257)
    val_data = torch.arange(256, dtype=torch.long).add(1).remainder(257)
    rng_before = torch.get_rng_state().clone()

    first = estimate_loss(
        model, train_data, val_data, batch_size=2, block_size=16,
        device=torch.device("cpu"), eval_iters=4, seed=2026,
    )
    rng_after_first = torch.get_rng_state().clone()
    second = estimate_loss(
        model, train_data, val_data, batch_size=2, block_size=16,
        device=torch.device("cpu"), eval_iters=4, seed=2026,
    )

    assert first == second
    assert torch.equal(rng_before, rng_after_first)
    assert torch.equal(rng_before, torch.get_rng_state())


def _tiny_checkpoint_parts():
    config = ModelConfig(vocab_size=257, block_size=16, n_layer=1, n_head=2, n_embd=16)
    model = ArtemisLM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    return model, optimizer


def test_save_checkpoint_round_trips_after_atomic_replace(tmp_path: Path) -> None:
    model, optimizer = _tiny_checkpoint_parts()
    checkpoint = tmp_path / "checkpoint.pt"

    save_checkpoint(
        checkpoint,
        model,
        optimizer,
        ByteTokenizer(),
        step=7,
        losses={"train": 1.25, "val": 1.5},
        response_only=False,
        run_metadata={"lineage": {"origin": "test"}, "runs": []},
        losses_at_step=5,
    )

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert payload["format_version"] == 3
    assert payload["step"] == 7
    assert payload["losses_at_step"] == 5
    assert payload["losses"]["val"] == pytest.approx(1.5)
    assert payload["tokenizer"] == ByteTokenizer.name
    assert not list(tmp_path.glob(".checkpoint.pt.*.tmp"))


def test_failed_checkpoint_write_preserves_previous_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, optimizer = _tiny_checkpoint_parts()
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"previous-good-checkpoint")

    def interrupted_save(_payload, stream) -> None:
        stream.write(b"partial-checkpoint")
        raise OSError("simulated disk failure")

    monkeypatch.setattr(train_module.torch, "save", interrupted_save)
    with pytest.raises(OSError, match="simulated disk failure"):
        save_checkpoint(
            checkpoint,
            model,
            optimizer,
            ByteTokenizer(),
            step=8,
            losses={"train": 1.0, "val": 1.1},
            response_only=False,
            run_metadata={"lineage": {"origin": "test"}, "runs": []},
        )

    assert checkpoint.read_bytes() == b"previous-good-checkpoint"
    assert not list(tmp_path.glob(".checkpoint.pt.*.tmp"))


def make_args(tmp_path: Path) -> Namespace:
    train = tmp_path / "train.txt"
    validation = tmp_path / "validation.txt"
    manifest = tmp_path / "manifest.json"
    train.write_text("training data", encoding="utf-8")
    validation.write_text("held-out data", encoding="utf-8")
    manifest.write_text('{"source":"test"}', encoding="utf-8")
    return Namespace(
        data=train,
        validation_data=validation,
        data_manifest=manifest,
        phase="foundation_pretraining",
        seed=1337,
        steps=20,
        learning_rate=0.0003,
        min_learning_rate=0.0,
        lr_schedule="constant",
        warmup_steps=0,
        allow_data_change=False,
        batch_size=4,
        response_only=False,
    )


def test_checkpoint_metadata_records_random_init_and_dataset_hashes(tmp_path: Path):
    args = make_args(tmp_path)
    metadata = build_checkpoint_metadata(
        args=args,
        config=ModelConfig(vocab_size=257, block_size=16, n_layer=1, n_head=2, n_embd=16),
        tokenizer=ByteTokenizer(),
        device=torch.device("cpu"),
        start_step=0,
        resume_payload=None,
        parent_checkpoint_sha256=None,
    )
    assert metadata["lineage"]["origin"] == "random_initialization"
    assert metadata["lineage"]["initialization_seed"] == 1337
    assert len(metadata["runs"]) == 1
    run = metadata["runs"][0]
    assert run["phase"] == "foundation_pretraining"
    assert len(run["train_text_sha256"]) == 64
    assert len(run["validation_text_sha256"]) == 64
    assert len(run["data_manifest_sha256"]) == 64
    assert run["parent_checkpoint_sha256"] is None
    assert run["source_commit"] is None or isinstance(run["source_commit"], str)


def test_checkpoint_metadata_preserves_root_lineage_on_resume(tmp_path: Path):
    args = make_args(tmp_path)
    config = ModelConfig(vocab_size=257, block_size=16, n_layer=1, n_head=2, n_embd=16)
    original = build_checkpoint_metadata(
        args=args, config=config, tokenizer=ByteTokenizer(),
        device=torch.device("cpu"), start_step=0, resume_payload=None,
        parent_checkpoint_sha256=None,
    )
    args.phase = "instruction_sft"
    args.response_only = True
    resumed = build_checkpoint_metadata(
        args=args, config=config, tokenizer=ByteTokenizer(),
        device=torch.device("cpu"), start_step=20,
        resume_payload={"run_metadata": original, "response_only": False},
        parent_checkpoint_sha256="a" * 64,
    )
    assert resumed["lineage"] == original["lineage"]
    assert len(resumed["runs"]) == 2
    assert resumed["runs"][-1]["parent_checkpoint_sha256"] == "a" * 64
    assert resumed["runs"][-1]["objective_changed_from_parent"] is True


def test_legacy_checkpoint_metadata_is_not_mislabelled_as_random_init(tmp_path: Path):
    args = make_args(tmp_path)
    metadata = build_checkpoint_metadata(
        args=args,
        config=ModelConfig(vocab_size=257, block_size=16, n_layer=1, n_head=2, n_embd=16),
        tokenizer=ByteTokenizer(),
        device=torch.device("cpu"),
        start_step=20,
        resume_payload={"response_only": False},
        parent_checkpoint_sha256="b" * 64,
    )
    assert metadata["lineage"]["origin"] == "legacy_checkpoint_origin_unverified"
    assert metadata["lineage"]["initialization_seed"] is None


def test_resume_data_hashes_must_match(tmp_path: Path):
    from artemis.train import validate_resume_data

    args = make_args(tmp_path)
    expected = {
        "train_text_sha256": __import__("artemis.train", fromlist=["sha256_file"]).sha256_file(args.data),
        "validation_text_sha256": __import__("artemis.train", fromlist=["sha256_file"]).sha256_file(args.validation_data),
        "data_manifest_sha256": __import__("artemis.train", fromlist=["sha256_file"]).sha256_file(args.data_manifest),
    }
    payload = {"run_metadata": {"runs": [expected]}}
    assert validate_resume_data(args=args, resume_payload=payload) is False

    args.data.write_text("changed training data", encoding="utf-8")
    with pytest.raises(SystemExit, match="Refusing to silently change"):
        validate_resume_data(args=args, resume_payload=payload)


def test_resume_data_change_requires_explicit_override(tmp_path: Path):
    from artemis.train import sha256_file, validate_resume_data

    args = make_args(tmp_path)
    payload = {"run_metadata": {"runs": [{
        "train_text_sha256": "0" * 64,
        "validation_text_sha256": sha256_file(args.validation_data),
        "data_manifest_sha256": sha256_file(args.data_manifest),
    }]}}
    with pytest.raises(SystemExit, match="--allow-data-change"):
        validate_resume_data(args=args, resume_payload=payload)

    args.allow_data_change = True
    assert validate_resume_data(args=args, resume_payload=payload) is True


def test_cosine_learning_rate_warmup_and_decay():
    from artemis.train import learning_rate_at_step

    kwargs = {
        "start_step": 8000,
        "target_step": 16000,
        "base_learning_rate": 1e-4,
        "min_learning_rate": 1e-5,
        "warmup_steps": 100,
        "schedule": "cosine",
    }
    assert learning_rate_at_step(step=8001, **kwargs) > 1e-5
    assert learning_rate_at_step(step=8100, **kwargs) == pytest.approx(1e-4)
    assert learning_rate_at_step(step=16000, **kwargs) == pytest.approx(1e-5)



def test_load_tokens_jsonl_adds_eos_per_document(tmp_path: Path) -> None:
    import json
    from artemis.train import load_tokens

    training_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    training_docs = [
        {"text": "alpha beta gamma delta " * 10, "source": "one"},
        {"text": "epsilon zeta eta theta " * 10, "source": "two"},
    ]
    validation_docs = [{"text": "validation material " * 20}]
    training_path.write_text(
        "".join(json.dumps(row) + "\n" for row in training_docs),
        encoding="utf-8",
    )
    validation_path.write_text(
        "".join(json.dumps(row) + "\n" for row in validation_docs),
        encoding="utf-8",
    )

    train_ids, val_ids = load_tokens(
        training_path,
        validation_path,
        ByteTokenizer(),
        document_format="jsonl",
    )
    assert int((train_ids == ByteTokenizer.eos_token_id).sum()) == 2
    assert int((val_ids == ByteTokenizer.eos_token_id).sum()) == 1
    assert train_ids[-1].item() == ByteTokenizer.eos_token_id


def test_load_tokens_jsonl_rejects_missing_text_field(tmp_path: Path) -> None:
    import json
    from artemis.train import load_tokens

    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"body": "no text field"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain string field"):
        load_tokens(path, path, ByteTokenizer(), document_format="jsonl")

from argparse import Namespace
from pathlib import Path

import torch
import pytest

from artemis.config import ModelConfig
from artemis.tokenizer import ByteTokenizer
from artemis.train import build_checkpoint_metadata, get_batch


def test_get_batch_returns_shifted_sequences() -> None:
    data = torch.arange(100, dtype=torch.long)
    x, y = get_batch(data, batch_size=4, block_size=8, device=torch.device("cpu"))
    assert x.shape == (4, 8)
    assert y.shape == (4, 8)
    assert torch.equal(y[:, :-1], x[:, 1:])


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

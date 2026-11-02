from argparse import Namespace
from pathlib import Path

import torch

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

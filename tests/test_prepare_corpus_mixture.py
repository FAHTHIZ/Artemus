import json

import pytest

from artemis.prepare_corpus_mixture import prepare_mixture


def write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_mixture_is_deterministic_weighted_and_preserves_provenance(tmp_path):
    en_train = tmp_path / "en-train.jsonl"
    en_val = tmp_path / "en-val.jsonl"
    en_beta_train = tmp_path / "bn-train.jsonl"
    en_beta_val = tmp_path / "bn-val.jsonl"

    write_jsonl(en_train, [
        {"text": f"English document {i} " * 8, "provenance": {"dataset_id": "en"}}
        for i in range(20)
    ])
    write_jsonl(en_val, [
        {"text": f"English validation {i} " * 8, "provenance": {"dataset_id": "en"}}
        for i in range(10)
    ])
    write_jsonl(en_beta_train, [
        {"text": f"English document number {i} " * 8, "provenance": {"dataset_id": "bn"}}
        for i in range(20)
    ])
    write_jsonl(en_beta_val, [
        {"text": f"English validation {i} " * 8, "provenance": {"dataset_id": "bn"}}
        for i in range(10)
    ])

    sources = [
        {"source_id": "english", "train_path": en_train, "validation_path": en_val, "weight": 0.8},
        {"source_id": "english_beta", "train_path": en_beta_train, "validation_path": en_beta_val, "weight": 0.2},
    ]
    first = prepare_mixture(
        sources, tmp_path / "mix-a", train_documents=10,
        validation_documents=5, seed=21,
    )
    second = prepare_mixture(
        sources, tmp_path / "mix-b", train_documents=10,
        validation_documents=5, seed=21,
    )

    assert first["train"] == second["train"]
    assert first["validation"] == second["validation"]
    assert first["train"]["per_source_documents"] == {"english": 8, "english_beta": 2}
    assert first["validation"]["per_source_documents"] == {"english": 4, "english_beta": 1}
    assert first["train_validation_hash_overlap"] == 0

    train_rows = [
        json.loads(line)
        for line in (tmp_path / "mix-a" / "train.jsonl").read_text().splitlines()
    ]
    assert all("provenance" in row and "mixture" in row for row in train_rows)
    assert {row["mixture"]["source_id"] for row in train_rows} == {"english", "english_beta"}


def test_mixture_removes_duplicates_across_sources_and_splits(tmp_path):
    shared = "A shared example that must never leak across train and validation splits."
    a_train, a_val, b_train, b_val = [tmp_path / f"{name}.jsonl" for name in ("at", "av", "bt", "bv")]
    write_jsonl(a_train, [{"text": shared}, {"text": "A unique training example for source alpha."}])
    write_jsonl(a_val, [{"text": "A unique validation document for source alpha."}])
    write_jsonl(b_train, [{"text": shared}, {"text": "A second unique training example for source beta."}])
    write_jsonl(b_val, [{"text": shared}, {"text": "A second unique validation document for source beta."}])

    result = prepare_mixture(
        [
            {"source_id": "a", "train_path": a_train, "validation_path": a_val, "weight": 1},
            {"source_id": "b", "train_path": b_train, "validation_path": b_val, "weight": 1},
        ],
        tmp_path / "mix",
        train_documents=10,
        validation_documents=10,
    )
    assert result["train_validation_hash_overlap"] == 0
    train_text = (tmp_path / "mix" / "train.txt").read_text()
    validation_text = (tmp_path / "mix" / "validation.txt").read_text()
    assert shared in train_text
    assert shared not in validation_text


def test_mixture_rejects_duplicate_or_invalid_source_weights(tmp_path):
    source = tmp_path / "source.jsonl"
    write_jsonl(source, [{"text": "sufficient sample material for token selection"}])
    with pytest.raises(ValueError, match="positive"):
        prepare_mixture(
            [{"source_id": "source", "train_path": source, "validation_path": source, "weight": 0}],
            tmp_path / "out",
        )

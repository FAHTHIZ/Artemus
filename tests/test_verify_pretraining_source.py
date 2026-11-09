import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def load_verifier():
    path = Path(__file__).resolve().parents[1] / "scripts" / "verify_pretraining_source.py"
    spec = importlib.util.spec_from_file_location("verify_pretraining_source", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_source(tmp_path, *, language="en", score=0.95):
    rows = []
    for split, count in (("train", 3), ("validation", 2)):
        path = tmp_path / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as stream:
            for index in range(count):
                text = f"English training example {split} {index} with useful context."
                provenance = {
                    "language": language,
                    "language_score": score,
                    "content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                }
                stream.write(json.dumps({"text": text, "provenance": provenance}) + "\n")
                rows.append((split, provenance))
    provenance_path = tmp_path / "provenance.jsonl"
    provenance_path.write_text(
        "".join(json.dumps({"language": language, "language_score": score}) + "\n" for _ in rows),
        encoding="utf-8",
    )
    manifest = {
        "dataset_id": "example/english-source",
        "dataset_revision": "a" * 40,
        "upstream_split": "train",
        "accepted_documents": len(rows),
        "counts": {"accepted_train": 3, "accepted_validation": 2},
        "language_policy": {
            "required_language": "en",
            "minimum_language_score": 0.70,
        },
        "files": {
            "train_jsonl": "train.jsonl",
            "validation_jsonl": "validation.jsonl",
            "provenance_jsonl": "provenance.jsonl",
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_verifier_accepts_complete_english_source_and_matches_counts(tmp_path):
    verifier = load_verifier()
    write_source(tmp_path)
    result = verifier.verify_prepared_source(
        tmp_path,
        minimum_language_score=0.70,
        expected_dataset_id="example/english-source",
    )
    assert result["verified"] is True
    assert result["accepted_documents"] == 5
    assert result["train_documents"] == 3
    assert result["validation_documents"] == 2


@pytest.mark.parametrize(("language", "score"), [("bn", 0.99), ("en", 0.40), ("", 0.99)])
def test_verifier_rejects_non_english_or_low_confidence_source(tmp_path, language, score):
    verifier = load_verifier()
    write_source(tmp_path, language=language, score=score)
    with pytest.raises(ValueError):
        verifier.verify_prepared_source(tmp_path, minimum_language_score=0.70)


def test_verifier_rejects_incomplete_provenance_or_split_counts(tmp_path):
    verifier = load_verifier()
    write_source(tmp_path)
    (tmp_path / "validation.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="missing or empty"):
        verifier.verify_prepared_source(tmp_path)


def test_verifier_rejects_wrong_dataset_identity(tmp_path):
    verifier = load_verifier()
    write_source(tmp_path)
    with pytest.raises(ValueError, match="Dataset ID mismatch"):
        verifier.verify_prepared_source(
            tmp_path,
            expected_dataset_id="example/wrong-source",
        )

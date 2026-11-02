import json
from pathlib import Path

from artemis.prepare_capability_data import (
    build_gsm8k_examples,
    build_mbpp_examples,
    git_blob_sha,
)


def test_gsm8k_train_rows_are_deterministically_split_and_formatted():
    rows = [
        {"question": f"Math question {i}", "answer": f"Reasoning\n#### {i}"}
        for i in range(100)
    ]
    train_a, val_a = build_gsm8k_examples(rows)
    train_b, val_b = build_gsm8k_examples(rows)
    assert train_a == train_b
    assert val_a == val_b
    assert len(train_a) + len(val_a) == 100
    assert train_a and val_a
    assert "<|assistant|>" in train_a[0]
    assert "<|end|>" in train_a[0]


def test_mbpp_ingests_only_official_train_and_validation_task_ids():
    rows = [
        {"task_id": 601, "text": "add two integers", "code": "def add(a,b): return a+b",
         "test_list": ["assert add(1,2)==3"], "test_setup_code": ""},
        {"task_id": 511, "text": "multiply two integers", "code": "def mul(a,b): return a*b",
         "test_list": ["assert mul(2,3)==6"], "test_setup_code": ""},
        {"task_id": 11, "text": "protected test", "code": "pass", "test_list": []},
        {"task_id": 2, "text": "prompt example", "code": "pass", "test_list": []},
    ]
    train, validation, counts = build_mbpp_examples(rows)
    assert len(train) == 1
    assert len(validation) == 1
    assert counts == {"train": 1, "validation": 1, "excluded": 2}
    assert "add two integers" in train[0]
    assert "assert add(1,2)==3" in train[0]
    assert "protected test" not in "".join(train + validation)


def test_mbpp_rejects_duplicate_task_ids():
    row = {"task_id": 601, "text": "task", "code": "pass", "test_list": []}
    import pytest
    with pytest.raises(ValueError, match="Duplicate MBPP"):
        build_mbpp_examples([row, row])


def test_git_blob_sha_matches_known_vector(tmp_path: Path):
    path = tmp_path / "blob.txt"
    path.write_text("hello\n", encoding="utf-8")
    assert git_blob_sha(path) == "ce013625030ba8dba906f756967f9e9ca394464a"

import json
from pathlib import Path

from artemis.prepare_capability_data import (
    build_evaluation_records,
    build_gsm8k_examples,
    build_mbpp_examples,
    git_blob_sha,
    prepare_mixture,
    sha256_file,
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
    diagnostic = build_evaluation_records(rows, [])
    assert len(train_a) + len(val_a) + len(diagnostic) == 100
    assert train_a and val_a and diagnostic
    all_monitor_text = "".join(train_a + val_a)
    eval_questions = {
        row["prompt"].splitlines()[-1] for row in diagnostic
    }
    assert all(question not in all_monitor_text for question in eval_questions)
    assert "<|assistant|>" in train_a[0]
    assert "<|end|>" in train_a[0]


def test_mbpp_ingests_only_official_train_and_validation_task_ids():
    rows = [
        {"task_id": 601, "text": "add two integers", "code": "def add(a,b): return a+b",
         "test_list": ["assert add(1,2)==3"], "test_setup_code": ""},
        {"task_id": 511, "text": "multiply two integers", "code": "def mul(a,b): return a*b",
         "test_list": ["assert mul(2,3)==6"], "test_setup_code": ""},
        {"task_id": 556, "text": "diagnostic task", "code": "def square(x): return x*x",
         "test_list": ["assert square(3)==9"], "test_setup_code": ""},
        {"task_id": 11, "text": "protected test", "code": "pass", "test_list": []},
        {"task_id": 2, "text": "prompt example", "code": "pass", "test_list": []},
    ]
    train, validation, counts = build_mbpp_examples(rows)
    assert len(train) == 1
    assert len(validation) == 1
    assert counts == {"train": 1, "validation": 1, "evaluation": 1, "excluded": 2}
    assert "add two integers" in train[0]
    assert "assert add(1,2)==3" in train[0]
    assert "protected test" not in "".join(train + validation)
    diagnostic = build_evaluation_records([], rows)
    assert {row["task_id"] for row in diagnostic} == {556}


def test_mbpp_rejects_duplicate_task_ids():
    row = {"task_id": 601, "text": "task", "code": "pass", "test_list": []}
    import pytest
    with pytest.raises(ValueError, match="Duplicate MBPP"):
        build_mbpp_examples([row, row])


def test_git_blob_sha_matches_known_vector(tmp_path: Path):
    path = tmp_path / "blob.txt"
    path.write_text("hello\n", encoding="utf-8")
    assert git_blob_sha(path) == "ce013625030ba8dba906f756967f9e9ca394464a"


def test_prepare_mixture_writes_hashed_structured_evaluation_file(tmp_path: Path):
    oasst_train = tmp_path / "oasst-train.txt"
    oasst_validation = tmp_path / "oasst-validation.txt"
    gsm_file = tmp_path / "gsm8k.jsonl"
    mbpp_file = tmp_path / "mbpp.jsonl"
    output_dir = tmp_path / "output"

    oasst_train.write_text("training conversation", encoding="utf-8")
    oasst_validation.write_text("validation conversation", encoding="utf-8")
    gsm_rows = [
        {"question": f"Compute {i} + 1.", "answer": f"Reasoning\n#### {i + 1}"}
        for i in range(100)
    ]
    mbpp_rows = [
        {"task_id": 511, "text": "monitor task", "code": "def f(a,b): return a*b",
         "test_list": ["assert f(2, 3) == 6"], "test_setup_code": ""},
        {"task_id": 556, "text": "diagnostic task", "code": "def square(x): return x*x",
         "test_list": ["assert square(3) == 9"], "test_setup_code": ""},
        {"task_id": 601, "text": "training task", "code": "pass", "test_list": []},
    ]
    gsm_file.write_text(
        "".join(json.dumps(row) + "\n" for row in gsm_rows),
        encoding="utf-8",
    )
    mbpp_file.write_text(
        "".join(json.dumps(row) + "\n" for row in mbpp_rows),
        encoding="utf-8",
    )

    manifest = prepare_mixture(
        oasst_train=oasst_train,
        oasst_validation=oasst_validation,
        gsm8k_train=gsm_file,
        mbpp_full=mbpp_file,
        output_dir=output_dir,
        verify_hashes=False,
    )
    evaluation_path = output_dir / "evaluation.jsonl"
    assert evaluation_path.is_file()
    assert manifest["evaluation_examples"] > 0
    assert manifest["evaluation_sha256"] == sha256_file(evaluation_path)
    records = [json.loads(line) for line in evaluation_path.read_text(encoding="utf-8").splitlines()]
    assert any(row["task_type"] == "math" for row in records)
    assert any(row["task_type"] == "python" and row["task_id"] == 556 for row in records)
    assert not any(row.get("task_id") in {511, 601} for row in records)

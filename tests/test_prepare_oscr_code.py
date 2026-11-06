import hashlib
import json
from collections.abc import Mapping
from typing import Any

from artemis.prepare_oscr_code import curate_oscr_files, repository_split, write_corpus


def make_file(content: str, path: str, block: str, row: int) -> dict[str, Any]:
    return {
        "path": path,
        "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "language": "Python",
        "lines": content.count("\n") + 1,
        "block": block,
        "row": row,
    }


def test_repository_split_is_deterministic_and_grouped() -> None:
    first = repository_split("https://github.com/example/project", validation_percent=10)
    assert first == repository_split("https://github.com/example/project", validation_percent=10)


def test_curator_keeps_only_allowlisted_verified_python_files() -> None:
    code = "def add(left, right):\n    return left + right\n"
    vendor = "print('not used')\n"
    rows: dict[tuple[str, int], Mapping[str, Any]] = {
        ("blocks/00001.parquet", 0): {
            "sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(),
            "language": "Python",
            "content": code,
        },
        ("blocks/00001.parquet", 1): {
            "sha256": hashlib.sha256(vendor.encode("utf-8")).hexdigest(),
            "language": "Python",
            "content": vendor,
        },
    }
    manifests = [
        {
            "format": "oscr-script-manifest/1",
            "address": "https://github.com/example/allowed",
            "commit": "abc123",
            "license": "MIT",
            "license_confirmed_by": "LICENSE file",
            "files": [
                make_file(code, "src/add.py", "blocks/00001.parquet", 0),
                make_file(vendor, "vendor/ignored.py", "blocks/00001.parquet", 1),
            ],
        },
        {
            "format": "oscr-script-manifest/1",
            "address": "https://github.com/example/restricted",
            "commit": "def456",
            "license": "GPL-3.0-only",
            "license_confirmed_by": "COPYING file",
            "files": [make_file(code, "src/add.py", "blocks/00001.parquet", 0)],
        },
    ]

    train, validation, counts = curate_oscr_files(
        manifests,
        read_row=lambda block, row: rows[(block, row)],
        allowed_spdx={"MIT"},
        validation_percent=0,
        min_bytes=1,
    )
    assert len(train) == 1
    assert validation == []
    assert train[0]["provenance"]["license"] == "MIT"
    assert train[0]["provenance"]["repository_commit"] == "abc123"
    assert counts["accepted_train"] == 1
    assert counts["skipped_path_filter"] == 1
    assert counts["rejected_license_not_allowlisted"] == 1


def test_curator_rejects_hash_mismatch_and_secret_pattern() -> None:
    content = "def value():\n    return 42\n"
    secret = "TOKEN = 'ghp_abcdefghijklmnopqrstuvwxyz1234567890'\n"
    manifests = [{
        "address": "https://github.com/example/project",
        "commit": "abc123",
        "license": "MIT",
        "license_confirmed_by": "LICENSE",
        "files": [
            {**make_file(content, "wrong.py", "blocks/a.parquet", 0), "sha256": "0" * 64},
            make_file(secret, "secret.py", "blocks/b.parquet", 0),
        ],
    }]
    rows = {
        ("blocks/a.parquet", 0): {
            "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "language": "Python",
            "content": content,
        },
        ("blocks/b.parquet", 0): {
            "sha256": hashlib.sha256(secret.encode("utf-8")).hexdigest(),
            "language": "Python",
            "content": secret,
        },
    }
    train, validation, counts = curate_oscr_files(
        manifests,
        read_row=lambda block, row: rows[(block, row)],
        allowed_spdx={"MIT"},
        validation_percent=0,
        min_bytes=1,
    )
    assert train == [] and validation == []
    assert counts["rejected_content_hash_mismatch"] == 1
    assert counts["rejected_secret_or_contact_pattern"] == 1


def test_write_corpus_preserves_source_lineage(tmp_path) -> None:
    content = "def ping():\n    return 'pong'\n"
    example = {
        "text": content,
        "provenance": {
            "repository": "https://github.com/example/project",
            "repository_commit": "abc123",
            "license": "MIT",
            "license_confirmed_by": "LICENSE",
            "path": "src/ping.py",
            "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "language": "Python",
            "lines": 2,
            "block": "blocks/00001.parquet",
            "row": 0,
            "split": "train",
        },
    }
    manifest = write_corpus(
        tmp_path,
        [example],
        [],
        {"accepted_train": 1, "accepted_total": 1},
        dataset_revision="b0e3fadb276678880f7f3d764308fbebd3bcd044",
        allowed_spdx={"MIT"},
        source_manifests=[{"path": "manifests/ab/repo.json", "sha256": "a" * 64}],
    )
    assert manifest["dataset_revision"].startswith("b0e3")
    record = json.loads((tmp_path / "train.jsonl").read_text().splitlines()[0])
    assert record["provenance"]["repository_commit"] == "abc123"
    assert (tmp_path / "train.txt").is_file()

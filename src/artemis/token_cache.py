"""Disk-backed token cache for large, independently curated JSONL documents.

The cache is a flat int32 token stream with one EOS token per source document.
PyTorch memory-maps the file, so random training batches page data in as needed
instead of expanding the entire corpus into Python integers and RAM tensors.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from artemis.tokenizer import ByteTokenizer, SentencePieceTokenizer


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tokenizer_identity(tokenizer: ByteTokenizer | SentencePieceTokenizer) -> dict[str, Any]:
    model_sha = (
        hashlib.sha256(tokenizer.serialized_model).hexdigest()
        if isinstance(tokenizer, SentencePieceTokenizer)
        else None
    )
    return {"name": tokenizer.name, "model_sha256": model_sha}


def build_jsonl_token_cache(
    source_path: Path,
    tokenizer: ByteTokenizer | SentencePieceTokenizer,
    cache_dir: Path,
    *,
    text_field: str = "text",
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Create or reuse a verified memory-mapped token stream for a JSONL split."""
    if not source_path.is_file():
        raise FileNotFoundError(f"JSONL source does not exist: {source_path}")
    if not text_field.strip():
        raise ValueError("text_field must not be empty")

    source_sha = sha256_file(source_path)
    tokenizer_info = _tokenizer_identity(tokenizer)
    identity = {
        "format_version": 1,
        "source_sha256": source_sha,
        "tokenizer": tokenizer_info,
        "text_field": text_field,
        "token_dtype": "little-endian-int32",
        "eos_per_document": True,
    }
    cache_key = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    cache_dir.mkdir(parents=True, exist_ok=True)
    token_path = cache_dir / f"{cache_key}.tokens.bin"
    metadata_path = cache_dir / f"{cache_key}.tokens.json"

    if token_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected_bytes = int(metadata.get("token_count", -1)) * np.dtype("<i4").itemsize
        if (
            all(metadata.get(key) == value for key, value in identity.items())
            and token_path.stat().st_size == expected_bytes
            and metadata.get("cache_sha256") == sha256_file(token_path)
            and int(metadata.get("token_count", 0)) > 0
        ):
            mapped = torch.from_file(
                str(token_path),
                shared=False,
                size=int(metadata["token_count"]),
                dtype=torch.int32,
            )
            return mapped, metadata

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{cache_key}.",
        suffix=".tmp",
        dir=cache_dir,
    )
    token_count = 0
    try:
        with os.fdopen(descriptor, "wb") as output, source_path.open(
            "r", encoding="utf-8"
        ) as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSONL at {source_path}:{line_number}"
                    ) from exc
                if not isinstance(row, dict) or not isinstance(row.get(text_field), str):
                    raise ValueError(
                        f"JSONL row at {source_path}:{line_number} must contain "
                        f"string field {text_field!r}"
                    )
                document = row[text_field].strip()
                if not document:
                    continue
                token_ids = tokenizer.encode(document, add_eos=True)
                if token_ids:
                    np.asarray(token_ids, dtype="<i4").tofile(output)
                    token_count += len(token_ids)

        if token_count < 40:
            raise ValueError(
                f"Tokenized JSONL split is too short (at least 40 tokens): {source_path}"
            )
        temporary_path = Path(temporary_name)
        cache_sha = sha256_file(temporary_path)
        os.replace(temporary_path, token_path)
        metadata = {
            **identity,
            "token_count": token_count,
            "cache_sha256": cache_sha,
            "cache_file": token_path.name,
        }
        metadata_tmp = metadata_path.with_name(metadata_path.name + ".tmp")
        metadata_tmp.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(metadata_tmp, metadata_path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise

    mapped = torch.from_file(
        str(token_path),
        shared=False,
        size=token_count,
        dtype=torch.int32,
    )
    return mapped, metadata

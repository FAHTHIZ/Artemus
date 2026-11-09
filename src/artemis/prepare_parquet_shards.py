"""Prepare pinned local Parquet shards with the standard Artemus provenance pipeline.

Streaming very large Hugging Face datasets through ``datasets`` can exhaust RAM on
small machines. This entry point reads already-downloaded, revision-pinned Parquet
shards in bounded row batches and feeds them to :func:`prepare_stream`, so every
language, quality, privacy, deduplication and provenance gate is identical.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator, Sequence

from artemis.prepare_pretraining_data import prepare_stream

DEFAULT_COLUMNS = (
    "text", "id", "url", "date", "dump", "language", "language_score",
    "token_count", "score", "int_score",
)


def iter_parquet_records(
    paths: Sequence[Path],
    *,
    columns: Sequence[str] = DEFAULT_COLUMNS,
    batch_size: int = 1024,
    min_prefilter_score: float | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield row dictionaries from Parquet shards using bounded memory.

    ``min_prefilter_score`` only skips rows early for speed; the authoritative
    quality gate is still applied (and counted) inside ``prepare_stream``.
    """
    import pyarrow.parquet as pq

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    for path in paths:
        parquet = pq.ParquetFile(path)
        available = set(parquet.schema_arrow.names)
        selected = [name for name in columns if name in available]
        if "text" not in selected:
            raise ValueError(f"{path} has no 'text' column")
        for batch in parquet.iter_batches(batch_size=batch_size, columns=selected):
            for row in batch.to_pylist():
                if min_prefilter_score is not None:
                    score = row.get("score")
                    if not isinstance(score, (int, float)) or score < min_prefilter_score:
                        continue
                yield row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, nargs="+", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config")
    parser.add_argument("--split", default="train")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--license", required=True, dest="license_id")
    parser.add_argument("--license-review-note", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-documents", type=int, default=20_000)
    parser.add_argument("--validation-percent", type=float, default=1.0)
    parser.add_argument("--min-chars", type=int, default=300)
    parser.add_argument("--max-chars", type=int, default=40_000)
    parser.add_argument("--min-alpha-ratio", type=float, default=0.35)
    parser.add_argument("--min-language-score", type=float, default=0.70)
    parser.add_argument("--min-quality-score", type=float)
    parser.add_argument("--batch-size", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in args.parquet:
        if not path.is_file():
            raise SystemExit(f"Parquet shard does not exist: {path}")
    records = iter_parquet_records(
        args.parquet,
        batch_size=args.batch_size,
        min_prefilter_score=None,
    )
    manifest = prepare_stream(
        records,
        dataset_id=args.dataset,
        dataset_revision=args.revision,
        dataset_config=args.config,
        dataset_split=args.split,
        license_id=args.license_id,
        license_review_note=args.license_review_note,
        output_dir=args.output_dir,
        max_documents=args.max_documents,
        validation_percent=args.validation_percent,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        min_alpha_ratio=args.min_alpha_ratio,
        min_language_score=args.min_language_score,
        min_quality_score=args.min_quality_score,
    )
    manifest["parquet_shards"] = [path.name for path in args.parquet]
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()

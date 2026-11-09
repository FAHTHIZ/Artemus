import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from artemis.prepare_parquet_shards import iter_parquet_records


def _write(path, rows):
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=3)


def test_iter_parquet_records_reads_all_rows_in_batches(tmp_path):
    rows = [{"text": f"doc {i}", "id": str(i), "score": float(i % 5), "extra": i} for i in range(10)]
    _write(tmp_path / "a.parquet", rows[:6])
    _write(tmp_path / "b.parquet", rows[6:])
    out = list(iter_parquet_records([tmp_path / "a.parquet", tmp_path / "b.parquet"], batch_size=2))
    assert [row["id"] for row in out] == [str(i) for i in range(10)]
    assert "extra" not in out[0]


def test_iter_parquet_records_optional_prefilter(tmp_path):
    rows = [{"text": "t", "score": s} for s in (1.0, 3.6, None, 4.2)]
    _write(tmp_path / "a.parquet", rows)
    out = list(iter_parquet_records([tmp_path / "a.parquet"], min_prefilter_score=3.5))
    assert [row["score"] for row in out] == [3.6, 4.2]


def test_iter_parquet_records_requires_text(tmp_path):
    _write(tmp_path / "a.parquet", [{"body": "x"}])
    with pytest.raises(ValueError, match="text"):
        list(iter_parquet_records([tmp_path / "a.parquet"]))

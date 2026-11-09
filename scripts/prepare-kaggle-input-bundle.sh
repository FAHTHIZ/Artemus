#!/usr/bin/env bash
set -euo pipefail

BUNDLE_ROOT="${1:-.kaggle-input-bundle}"
PREP_ROOT="${2:-.kaggle-prepared-inputs}"
INPUT_DATASET="${KAGGLE_INPUT_DATASET:-}"
if [[ -z "$INPUT_DATASET" ]]; then
  echo "::error::KAGGLE_INPUT_DATASET must be set to the private Kaggle input dataset ID."
  exit 1
fi

rm -rf "$PREP_ROOT" "$BUNDLE_ROOT"
mkdir -p "$PREP_ROOT" "$BUNDLE_ROOT/prepared-artifacts" "$BUNDLE_ROOT/wheelhouse"

# Avoid the optional native Xet transfer backend in this streaming-preparation
# process. Keep a tested, mutually compatible data-ingestion stack instead of
# silently taking the newest Hub/fsspec/pyarrow combination.
export HF_HUB_DISABLE_XET=1
export HF_HUB_DISABLE_TELEMETRY=1
ulimit -c 0 || true

# Install only the project entry points plus pinned data-ingestion tooling on
# the internet-connected runner. This step prepares data; it does not train weights.
python -m pip install --no-deps --no-build-isolation -e .
python -m pip install \
  "numpy==1.26.4" \
  "fsspec[http]==2025.3.0" \
  "pyarrow==20.0.0" \
  "huggingface_hub==0.34.4" \
  "datasets==3.6.0"

# A native extension can abort during interpreter finalization after the
# complete manifest has been written. Only tolerate SIGABRT (134) when a
# separate verifier proves every split and provenance record is complete.
run_english_preparation() {
  local output_dir="$1"
  local expected_dataset="$2"
  shift 2
  local command_status=0

  set +e
  "$@"
  command_status=$?
  set -e

  python scripts/verify_pretraining_source.py "$output_dir" \
    --minimum-language-score 0.70 \
    --expected-dataset-id "$expected_dataset"

  if [[ "$command_status" -ne 0 ]]; then
    if [[ "$command_status" -eq 134 ]]; then
      echo "::warning::The source output passed manifest, split, hash, provenance and English-language verification despite a native interpreter-shutdown abort (exit 134)."
    else
      echo "::error::English source preparation failed with exit code $command_status."
      return "$command_status"
    fi
  fi
}

FINEWEB_NOTE="ODC-By-1.0 research pilot; Common Crawl terms and residual PII caveats require separate review before redistribution."
# FineWeb-Edu is read from one pinned Parquet shard in bounded row batches.
# Streaming hundreds of thousands of rows to reach the score >= 3.5 tier can
# exhaust runner memory; the local reader applies identical gates.
FINEWEB_EDU_REVISION="87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
FINEWEB_EDU_SHARD="sample/10BT/000_00000.parquet"
FINEWEB_EDU_SHARD_PATH="$(python - "$FINEWEB_EDU_REVISION" "$FINEWEB_EDU_SHARD" "$PREP_ROOT/raw" <<'PY'
import sys
from huggingface_hub import hf_hub_download
revision, filename, local_dir = sys.argv[1:4]
print(hf_hub_download(
    "HuggingFaceFW/fineweb-edu", filename, repo_type="dataset",
    revision=revision, local_dir=local_dir,
))
PY
)"
run_english_preparation "$PREP_ROOT/fineweb_edu" "HuggingFaceFW/fineweb-edu" \
  artemus-prepare-parquet \
  --parquet "$FINEWEB_EDU_SHARD_PATH" \
  --dataset HuggingFaceFW/fineweb-edu \
  --config sample-10BT \
  --split train \
  --revision "$FINEWEB_EDU_REVISION" \
  --license ODC-By-1.0 \
  --license-review-note "$FINEWEB_NOTE" \
  --output-dir "$PREP_ROOT/fineweb_edu" \
  --max-documents 80000 \
  --validation-percent 1.0 \
  --min-chars 300 \
  --max-chars 40000 \
  --min-language-score 0.70 \
  --min-quality-score 3.5
rm -rf "$PREP_ROOT/raw"

# General English web text is the foundation base; FineWeb-Edu is a smaller
# quality-weighted supplement rather than the whole language distribution.
FINEWEB_GENERAL_REVISION="$(python - <<'PY'
from huggingface_hub import HfApi
revision = HfApi().dataset_info("HuggingFaceFW/fineweb").sha
if not revision or len(revision) < 7:
    raise SystemExit("Could not resolve FineWeb revision")
print(revision)
PY
)"
FINEWEB_GENERAL_NOTE="ODC-By-1.0 research pilot; Common Crawl terms and residual PII/content caveats apply. Local pattern screening is not a complete privacy audit."
run_english_preparation "$PREP_ROOT/fineweb_general" "HuggingFaceFW/fineweb" \
  artemus-prepare-hf \
  --dataset HuggingFaceFW/fineweb \
  --config sample-10BT \
  --split train \
  --revision "$FINEWEB_GENERAL_REVISION" \
  --license ODC-By-1.0 \
  --license-review-note "$FINEWEB_GENERAL_NOTE" \
  --output-dir "$PREP_ROOT/fineweb_general" \
  --max-documents 40000 \
  --validation-percent 1.0 \
  --min-chars 300 \
  --max-chars 40000 \
  --required-language en \
  --min-language-score 0.70 \
  --shuffle-seed 11337
python - "$PREP_ROOT/fineweb_general/manifest.json" "$FINEWEB_GENERAL_REVISION" <<'PY'
import json, sys
from pathlib import Path
path, revision = Path(sys.argv[1]), sys.argv[2]
data = json.loads(path.read_text(encoding="utf-8"))
data["resolved_revision"] = revision
path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY

AYA_REVISION="$(python - <<'PY'
from huggingface_hub import HfApi
revision = HfApi().dataset_info("CohereLabs/aya_dataset").sha
if not revision or len(revision) < 7:
    raise SystemExit("Could not resolve Aya dataset revision")
print(revision)
PY
)"
artemus-prepare-aya \
  --revision "$AYA_REVISION" \
  --split train \
  --output-dir "$PREP_ROOT/aya_sft" \
  --max-examples 10000 \
  --validation-percent 2.0 \
  --shuffle-seed 3337 \
  --language-codes eng
python - "$PREP_ROOT/aya_sft/manifest.json" "$AYA_REVISION" <<'PY'
import json, sys
from pathlib import Path
path, revision = Path(sys.argv[1]), sys.argv[2]
data = json.loads(path.read_text(encoding="utf-8"))
data["resolved_revision"] = revision
path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY

python - <<'PY'
import hashlib
from pathlib import Path
from urllib.request import urlretrieve
from artemis.prepare_oasst1 import SOURCE_URL, EXPECTED_SHA256
from artemis.prepare_capability_data import (
    GSM8K_URL, GSM8K_SHA256, MBPP_URL, MBPP_GIT_BLOB_SHA,
)
raw = Path(".kaggle-raw-capability-data")
raw.mkdir(parents=True, exist_ok=True)
files = [
    (SOURCE_URL, raw / "oasst1-ready.messages.jsonl.gz", EXPECTED_SHA256, "sha256"),
    (GSM8K_URL, raw / "gsm8k-train.jsonl", GSM8K_SHA256, "sha256"),
    (MBPP_URL, raw / "mbpp-full.jsonl", MBPP_GIT_BLOB_SHA, "git-blob"),
]
for url, path, expected, algorithm in files:
    if not path.is_file():
        print(f"Downloading pinned source: {path.name}")
        urlretrieve(url, path)
    data = path.read_bytes()
    if algorithm == "sha256":
        actual = hashlib.sha256(data).hexdigest()
    else:
        actual = hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()
    if actual != expected:
        raise SystemExit(f"Checksum mismatch for {path.name}: {actual}")
print("Pinned OASST1, GSM8K train, and MBPP source checksums verified.")
PY

python -m artemis.prepare_oasst1 \
  --source .kaggle-raw-capability-data/oasst1-ready.messages.jsonl.gz \
  --output-dir "$PREP_ROOT/oasst1" \
  --max-examples 5000
python -m artemis.prepare_capability_data \
  --oasst-train "$PREP_ROOT/oasst1/train.txt" \
  --oasst-validation "$PREP_ROOT/oasst1/validation.txt" \
  --gsm8k-train .kaggle-raw-capability-data/gsm8k-train.jsonl \
  --mbpp-full .kaggle-raw-capability-data/mbpp-full.jsonl \
  --output-dir "$PREP_ROOT/capability_sft"

# Ship the exact source revision and only the filtered/prepared inputs. The
# upstream raw corpora stay out of the Kaggle bundle; each prepared directory
# contains its source manifest and provenance record.
python - "$BUNDLE_ROOT" "$GITHUB_SHA" "$PREP_ROOT" <<'PY'
import hashlib
import json
import sys
import zipfile
from pathlib import Path

bundle = Path(sys.argv[1])
commit = sys.argv[2]
prepared = Path(sys.argv[3])
repo = Path.cwd()

with zipfile.ZipFile(bundle / "artemus-source.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for relative in (Path("src"), Path("pyproject.toml"), Path("README.md"), Path("data/evaluation")):
        path = repo / relative
        if path.is_dir():
            for file in sorted(path.rglob("*")):
                if file.is_file():
                    archive.write(file, file.relative_to(repo).as_posix())
        elif path.is_file():
            archive.write(path, path.relative_to(repo).as_posix())

(bundle / "repository-commit.txt").write_text(commit + "\n", encoding="utf-8")
for name in ("fineweb_general", "fineweb_edu", "aya_sft", "oasst1", "capability_sft"):
    source = prepared / name
    if not source.is_dir() or not (source / "manifest.json").is_file():
        raise SystemExit(f"Prepared source is incomplete: {name}")
    target = bundle / "prepared-artifacts" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle / "prepared-artifacts" / f"{name}.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(source.rglob("*")):
            if file.is_file():
                archive.write(file, file.relative_to(source).as_posix())

(bundle / "wheelhouse").mkdir(exist_ok=True)
records = []
for path in sorted(p for p in bundle.rglob("*") if p.is_file()):
    records.append({
        "path": path.relative_to(bundle).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    })
manifest = {
    "schema_version": 1,
    "repository_commit": commit,
    "artifacts": records,
}
(bundle / "bundle-manifest.json").write_text(
    json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
)
PY

# Kaggle's Python 3.13 CPU runtime cannot fetch PyPI wheels if notebook
# internet is disabled, so include the matching wheel in the private bundle.
python -m pip download \
  --dest "$BUNDLE_ROOT/wheelhouse" \
  --only-binary=:all: \
  --python-version 3.13 \
  --implementation cp \
  --abi cp313 \
  --platform manylinux_2_27_x86_64 \
  --platform manylinux_2_28_x86_64 \
  "sentencepiece>=0.2.0"

python - <<'PY'
from pathlib import Path
wheelhouse = Path(".kaggle-input-bundle/wheelhouse")
wheels = list(wheelhouse.glob("sentencepiece-*-cp313-*.whl"))
if not wheels:
    raise SystemExit("No CPython 3.13 SentencePiece wheel was bundled.")
print("Bundled SentencePiece wheel:", wheels[0].name)
PY

cat > "$BUNDLE_ROOT/README.md" <<EOF
# Artemus private offline training inputs

Repository commit: $GITHUB_SHA

This private dataset contains the exact Artemus source archive, pinned and filtered training input splits, source manifests/provenance records, held-out evaluation data, and a CPython 3.13 SentencePiece wheel for Kaggle's offline runtime.

Upstream licence and data caveats are preserved in each prepared source manifest. This bundle is for the Artemus training notebook; do not make it public.
EOF

python - "$BUNDLE_ROOT/dataset-metadata.json" "$INPUT_DATASET" <<'PY'
import json, sys
from pathlib import Path
target, dataset_id = Path(sys.argv[1]), sys.argv[2]
payload = {
    "title": "Artemus Training Inputs",
    "id": dataset_id,
    "licenses": [{"name": "other"}],
    "description": (
        "Private, filtered and provenance-preserving source inputs and source code "
        "for reproducible Artemus training. Keep private; per-source licence details "
        "and review notes are included in manifests."
    ),
}
target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

echo "Offline training input bundle prepared:"
du -sh "$BUNDLE_ROOT"
python - "$BUNDLE_ROOT/bundle-manifest.json" <<'PY'
import json, sys
from pathlib import Path
m = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(json.dumps({
    "repository_commit": m["repository_commit"],
    "artifact_count": len(m["artifacts"]),
    "total_bytes": sum(item["size_bytes"] for item in m["artifacts"]),
}, indent=2))
PY

python - "$BUNDLE_ROOT/bundle-manifest.json" "$GITHUB_SHA" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
root, commit = Path(sys.argv[1]).parent, sys.argv[2]
records = []
for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != "bundle-manifest.json"):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    records.append({
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": digest,
    })
payload = {
    "schema_version": 1,
    "repository_commit": commit,
    "artifacts": records,
    "total_bytes": sum(item["size_bytes"] for item in records),
}
(root / "bundle-manifest.json").write_text(
    json.dumps(payload, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps({
    "repository_commit": commit,
    "artifact_count": len(records),
    "total_bytes": payload["total_bytes"],
}, indent=2))
PY

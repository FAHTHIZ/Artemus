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

# Install only the project entry points plus data-ingestion tooling on GitHub's
# internet-connected runner. No model training or model weights are produced here.
python -m pip install --upgrade pip
python -m pip install --no-deps --no-build-isolation -e .
python -m pip install "datasets>=3.0" "huggingface_hub>=0.26" "pyarrow>=18.0"

FINEWEB_NOTE="ODC-By-1.0 research pilot; Common Crawl terms and residual PII caveats require separate review before redistribution."
artemus-prepare-hf \
  --dataset HuggingFaceFW/fineweb-edu \
  --config sample-10BT \
  --split train \
  --revision 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9 \
  --license ODC-By-1.0 \
  --license-review-note "$FINEWEB_NOTE" \
  --output-dir "$PREP_ROOT/fineweb_edu" \
  --max-documents 20000 \
  --validation-percent 1.0 \
  --min-chars 300 \
  --max-chars 40000 \
  --shuffle-seed 1337

FINEWEB2_REVISION="$(python - <<'PY'
from huggingface_hub import HfApi
revision = HfApi().dataset_info("HuggingFaceFW/fineweb-2").sha
if not revision or len(revision) < 7:
    raise SystemExit("Could not resolve FineWeb2 revision")
print(revision)
PY
)"
FINEWEB2_NOTE="ODC-By-1.0 research pilot; Common Crawl terms and residual PII caveats apply. Additional pattern filtering is not a complete privacy audit."
artemus-prepare-hf \
  --dataset HuggingFaceFW/fineweb-2 \
  --config ben_Beng \
  --split train \
  --revision "$FINEWEB2_REVISION" \
  --license ODC-By-1.0 \
  --license-review-note "$FINEWEB2_NOTE" \
  --output-dir "$PREP_ROOT/fineweb2_bangla" \
  --max-documents 5000 \
  --validation-percent 1.0 \
  --min-chars 300 \
  --max-chars 40000 \
  --shuffle-seed 2337
python - "$PREP_ROOT/fineweb2_bangla/manifest.json" "$FINEWEB2_REVISION" <<'PY'
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
  --max-examples 1500 \
  --validation-percent 2.0 \
  --shuffle-seed 3337
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
for name in ("fineweb_edu", "fineweb2_bangla", "aya_sft", "oasst1", "capability_sft"):
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
    "title": "Artemus Offline Training Inputs",
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

"""Static integrity tests for user-facing training notebooks."""

import json
from pathlib import Path


def test_kaggle_training_notebook_is_valid_json_and_all_code_cells_compile() -> None:
    notebook_path = Path(__file__).resolve().parents[1] / "notebooks" / "kaggle-gpu-training.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    assert notebook.get("nbformat") == 4
    assert isinstance(notebook.get("cells"), list) and notebook["cells"]

    code_cells = [cell for cell in notebook["cells"] if cell.get("cell_type") == "code"]
    assert code_cells, "training notebook must contain executable code cells"
    for index, cell in enumerate(code_cells):
        source = cell.get("source", [])
        source_text = "".join(source) if isinstance(source, list) else str(source)
        compile(source_text, f"{notebook_path.name}:cell-{index}", "exec")


def test_kaggle_training_notebook_does_not_delete_existing_workspace() -> None:
    notebook_path = Path(__file__).resolve().parents[1] / "notebooks" / "kaggle-gpu-training.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )

    assert "shutil.rmtree(WORKDIR)" not in source
    assert "json.loads((ARTIFACT_ROOT / 'dialogue-eval' / 'summary.json').read_text())" in source


def test_kaggle_training_notebook_exports_complete_stage_provenance() -> None:
    notebook_path = Path(__file__).resolve().parents[1] / "notebooks" / "kaggle-gpu-training.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    export_cells = [
        cell for cell in notebook["cells"]
        if cell.get("cell_type") == "code" and cell.get("id") == "export-code"
    ]
    assert len(export_cells) == 1, "training notebook must have one identifiable export cell"
    source = "".join(export_cells[0].get("source", []))

    for expected in (
        '("fineweb-edu", DATA_DIR)',
        '("fineweb2-bangla", FINEWEB2_DIR)',
        '("foundation-mixture", MIX_DIR)',
        '("aya-instruction-sft", AYA_DIR)',
        '"run-metadata.json"',
        '"export-manifest.json"',
        '"repository_commit": REPO_COMMIT',
        "sha256_path",
    ):
        assert expected in source, f"training export is missing {expected!r}"

    assert 'if OSCR_MANIFEST_PATHS:' in source
    assert "provenance.jsonl" in source
    assert "shutil.rmtree" not in source


def test_kaggle_training_notebook_writes_valid_aya_manifest_newline() -> None:
    notebook_path = Path(__file__).resolve().parents[1] / "notebooks" / "kaggle-gpu-training.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    aya_cells = [
        cell for cell in notebook["cells"]
        if cell.get("cell_type") == "code" and cell.get("id") == "aya-code"
    ]
    assert len(aya_cells) == 1, "training notebook must have one Aya preparation cell"
    source = "".join(aya_cells[0].get("source", []))

    assert r"""json.dumps(aya_manifest, indent=2, ensure_ascii=False) + '\n'""" in source
    assert r"""json.dumps(aya_manifest, indent=2, ensure_ascii=False) + '\\n'""" not in source

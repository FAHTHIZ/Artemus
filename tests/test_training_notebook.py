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

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



def test_kaggle_training_notebook_resumes_foundation_state_and_rotates_samples() -> None:
    notebook_path = Path(__file__).resolve().parents[1] / "notebooks" / "kaggle-gpu-training.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code = {
        cell.get("id"): "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    }

    settings = code["settings"]
    preparation = code["stream"]
    tokenizer = code["tokenizer-code"]
    training = code["train-code"]
    aya = code["aya-code"]
    export = code["export-code"]

    assert "artemus-foundation-current.pt" in settings
    assert "response_only" in settings
    assert "tokenizer_model_b64" in settings
    assert "embedded_sha != external_sha" in settings
    assert "FINEWEB_SHUFFLE_SEED" in settings
    assert "--shuffle-seed" in preparation
    assert "HAS_FOUNDATION_STATE" in tokenizer
    assert "SentencePieceTokenizer.train" in tokenizer
    assert "--resume" in training
    assert "--allow-data-change" in training
    assert "continued_pretraining" in training
    assert "previous_step + TRAIN_STEPS" in training
    assert "--shuffle-seed" in aya
    assert "'training_cycle': CYCLE_INDEX + 1" in export
    assert "shutil.rmtree" not in "\n".join(code.values())


def test_recurring_kaggle_workflow_persists_private_foundation_state() -> None:
    workflow_path = (
        Path(__file__).resolve().parents[1]
        / ".github" / "workflows" / "continuous-kaggle-training.yml"
    )
    source = workflow_path.read_text(encoding="utf-8")

    for expected in (
        'cron: "0 3 * * 1"',
        "KAGGLE_USERNAME",
        "KAGGLE_KEY",
        "KAGGLE_STATE_DATASET",
        "kaggle kernels push",
        "kaggle kernels status",
        "kaggle kernels output",
        "kaggle datasets version",
        "artemus-foundation-current.pt",
        "artemus-latest-instruction.pt",
        "export-manifest.json",
    ):
        assert expected in source, f"recurring Kaggle workflow is missing {expected!r}"

    assert "timeout-minutes: 360" in source
    assert "cancel-in-progress: false" in source

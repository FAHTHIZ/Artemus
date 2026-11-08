"""Static integrity tests for user-facing training notebooks."""

import ast
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
    assert "artemus-dialogue-eval" in source
    assert "json.loads((ARTIFACT_ROOT / 'dialogue-eval' / 'summary.json').read_text())" in source
    assert "import hashlib" in source


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
    assert "STATE_MANIFEST" in settings
    assert "foundation_checkpoint_sha256" in settings
    assert "tokenizer_sha256" in settings
    assert "recorded_cycle != CYCLE_INDEX" in settings
    assert "FINEWEB_SHUFFLE_SEED" in settings
    assert "--shuffle-seed" in preparation
    assert "HAS_FOUNDATION_STATE" in tokenizer
    assert "SentencePieceTokenizer.train" in tokenizer
    assert "--resume" in training
    assert "--allow-data-change" in training
    assert "continued_pretraining" in training
    assert "previous_step + TRAIN_STEPS" in training
    assert "--shuffle-seed" in aya
    assert '"training_cycle": CYCLE_INDEX + 1' in export
    assert '"foundation_global_step": foundation_global_step' in export
    assert '"foundation_training_run_count": foundation_training_run_count' in export
    assert 'foundation_checkpoint_sha256 = sha256_path(foundation_checkpoint_export)' in export
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
        "NvidiaTeslaA100",
        "--dir-mode zip",
        "KAGGLE_KERNEL_SLUG",
        "export-manifest.json",
        "artemus-task-sft.pt",
        '"global_step": foundation_global_step',
        '"final_candidate_global_step"',
        "Foundation checkpoint hash does not match run metadata.",
        "foundation_run_count != training_cycle",
    ):
        assert expected in source, f"recurring Kaggle workflow is missing {expected!r}"

    assert "timeout-minutes: 360" in source
    assert "cancel-in-progress: false" in source


def test_gpu_notebook_contains_verified_task_sft_stage() -> None:
    notebook_path = Path(__file__).resolve().parents[1] / "notebooks" / "kaggle-gpu-training.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code = {
        cell.get("id"): "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    }
    assert "BLOCK_SIZE = 512" in code["settings"]
    assert "TASK_SFT_STEPS" in code["settings"]
    task = code["task-sft-code"]
    for expected in (
        "OASST1_SHA256",
        "GSM8K_SHA256",
        "MBPP_GIT_BLOB_SHA",
        "bytes([0])",
        "artemis.prepare_oasst1",
        "artemis.prepare_capability_data",
        "'--phase', 'task_sft'",
        "'--response-only'",
        "CHECKPOINT = TASK_CHECKPOINT",
    ):
        assert expected in task, f"task SFT stage is missing {expected!r}"
    export = code["export-code"]
    assert 'CHECKPOINT_ROOT / "artemus-task-sft.pt"' in export
    assert '("capability-sft", CAPABILITY_DIR)' in export
    assert '("oasst1", OASST_DIR)' in export


def test_kaggle_training_stages_configure_independent_checkpoint_cadence() -> None:
    notebook_path = Path(__file__).resolve().parents[1] / "notebooks" / "kaggle-gpu-training.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code = {
        cell.get("id"): "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    }

    assert "CHECKPOINT_INTERVAL = 100" in code["settings"]
    for cell_id in ("train-code", "oscr-code", "aya-code", "task-sft-code"):
        assert "--checkpoint-interval" in code[cell_id], (
            f"{cell_id} must checkpoint independently of its evaluation interval"
        )


def test_kaggle_workflow_promotes_task_tuned_candidate_not_aya_intermediate() -> None:
    workflow_path = (
        Path(__file__).resolve().parents[1]
        / ".github" / "workflows" / "continuous-kaggle-training.yml"
    )
    source = workflow_path.read_text(encoding="utf-8")
    assert 'instruction = root / "artemus-task-sft.pt"' in source
    assert "artemus-latest-aya.pt" in source
    assert "artemus-latest-instruction.pt" in source
    assert "cp verified-run/artemus-task-sft.pt kaggle-state-update/artemus-latest-instruction.pt" in source


def test_kaggle_gpu_workflow_uses_supported_accelerator_ids_and_uploads_provenance_dirs() -> None:
    workflow_path = (
        Path(__file__).resolve().parents[1]
        / ".github" / "workflows" / "continuous-kaggle-training.yml"
    )
    source = workflow_path.read_text(encoding="utf-8")
    assert "NvidiaTeslaT4" in source
    assert "NvidiaL4" in source
    assert "NvidiaTeslaA100" in source
    assert "NvidiaA100" not in source
    assert "--dir-mode zip" in source
    assert 'KAGGLE_KERNEL_SLUG" != "artemus-continuous-training"' in source


def test_machine_readable_curriculum_matches_continuous_training_cycle() -> None:
    curriculum_path = Path(__file__).resolve().parents[1] / "data" / "training-curriculum.json"
    curriculum = json.loads(curriculum_path.read_text(encoding="utf-8"))
    cycle = curriculum["continuous_cycle"]

    assert cycle["cadence"] == "weekly"
    assert cycle["foundation_checkpoint"] == "artemus-foundation-current.pt"
    assert cycle["tokenizer"] == "tokenizer.model; reused byte-identically across continued-pretraining cycles"
    assert cycle["per_cycle"]["foundation_steps"] == 2000
    assert cycle["per_cycle"]["multilingual_instruction_examples_max"] == 10000
    assert cycle["per_cycle"]["task_steps"] == 750
    assert cycle["per_cycle"]["final_candidate"] == "artemus-task-sft.pt"
    assert "OpenAssistant/oasst1" in cycle["per_cycle"]["task_sources"]
    assert "openai/grade-school-math" in cycle["per_cycle"]["task_sources"]
    assert "google-research-datasets/mbpp" in cycle["per_cycle"]["task_sources"]
    assert any("never an Aya or task-SFT candidate" in item for item in cycle["invariants"])
    assert "bucket 0 is optimizer validation" in cycle["evaluation_partition"]["gsm8k"]
    assert "bucket 1 is diagnostic evaluation" in cycle["evaluation_partition"]["gsm8k"]
    assert "511-555 are optimizer validation" in cycle["evaluation_partition"]["mbpp"]
    assert "556-600 are diagnostic evaluation" in cycle["evaluation_partition"]["mbpp"]
    assert "disjoint from both training and optimizer validation" in cycle["evaluation_partition"]["invariant"]

def test_kaggle_workflow_bootstraps_private_state_dataset_on_first_run() -> None:
    workflow_path = (
        Path(__file__).resolve().parents[1]
        / ".github" / "workflows" / "continuous-kaggle-training.yml"
    )
    source = workflow_path.read_text(encoding="utf-8")
    assert "kaggle datasets create -p .kaggle-state-bootstrap -q" in source
    assert '"licenses": [{"name": "other"}]' in source
    assert '"id": dataset_id' in source
    assert "KAGGLE_STATE_DATASET" in source
    assert "if metadata.get(\"id\") != expected:" in source
    assert "kaggle datasets create -p .kaggle-state-bootstrap -q --public" not in source


def test_gpu_notebook_runs_and_exports_capability_evaluation() -> None:
    notebook_path = Path(__file__).resolve().parents[1] / "notebooks" / "kaggle-gpu-training.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code = {
        cell.get("id"): "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    }
    assert "'artemus-capability-eval'" in code["eval-code"]
    assert "CAPABILITY_DIR / 'evaluation.jsonl'" in code["eval-code"]
    assert "--max-math-examples" in code["eval-code"]
    assert "--max-code-examples" in code["eval-code"]
    assert "capability-evaluation-summary.json" in code["export-code"]
    assert "capability-evaluation-results.jsonl" in code["export-code"]
    assert "capability-evaluation-input.jsonl" in code["export-code"]


def test_gpu_notebook_compares_all_training_stages_on_the_same_eval_set() -> None:
    notebook_path = Path(__file__).resolve().parents[1] / "notebooks" / "kaggle-gpu-training.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code = {
        cell.get("id"): "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    }
    for stage in ("foundation", "aya", "task-sft"):
        assert f"('{stage}', CHECKPOINT_ROOT" in code["eval-code"]
    assert 'for stage_name in ("foundation", "aya", "task-sft")' in code["export-code"]
    assert 'f"capability-evaluation-{stage_name}-summary.json"' in code["export-code"]
    assert "evaluation_data_hashes" in code["eval-code"]
    assert "capability-evaluation-comparison.json" in code["export-code"]


def test_kaggle_notebook_code_cells_preserve_line_boundaries_and_export_executes() -> None:
    notebook_path = Path(__file__).resolve().parents[1] / "notebooks" / "kaggle-gpu-training.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell.get("cell_type") == "code"]

    for cell in code_cells:
        source = cell.get("source", [])
        if isinstance(source, list):
            for line in source[:-1]:
                assert line.endswith("\n"), f"cell {cell.get('id')} has a joined source-line boundary"

    export_cells = [cell for cell in code_cells if cell.get("id") == "export-code"]
    assert len(export_cells) == 1
    export_source = "".join(export_cells[0]["source"])
    parsed = ast.parse(export_source)
    assert parsed.body, "export cell must contain executable statements, not only a comment"
    assert "EXPORT_ZIP = shutil.make_archive" in export_source
    assert "export-manifest.json" in export_source


def test_kaggle_workflows_support_api_token_and_preflight_authentication() -> None:
    root = Path(__file__).resolve().parents[1]
    for relative in (".github/workflows/validation.yml", ".github/workflows/continuous-kaggle-training.yml"):
        source = (root / relative).read_text(encoding="utf-8")
        assert "KAGGLE_API_TOKEN: ${{ secrets.KAGGLE_API_TOKEN }}" in source
        assert "kaggle quota" in source
        assert "unset KAGGLE_KEY" in source
        assert "KAGGLE_API_TOKEN" in source and "KAGGLE_KEY" in source


def test_initial_training_launcher_runs_fresh_and_preserves_first_foundation_checkpoint() -> None:
    workflow_path = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "validation.yml"
    source = workflow_path.read_text(encoding="utf-8")
    assert "needs: tests" in source
    assert "kaggle quota" in source
    assert '"dataset_sources": []' in source
    assert "actions/upload-artifact@v4" in source
    assert "artemus-initial-training-${{ github.run_id }}" in source
    assert "kaggle datasets version" in source
    assert "artemus-foundation-current.pt" in source
    assert "artemus-latest-instruction.pt" in source
    assert "state.json" in source
    assert source.index("kaggle kernels push") < source.index("kaggle datasets create")


def test_kaggle_username_normalizer_accepts_plain_quoted_field_and_json_values() -> None:
    import runpy

    normalize_username = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "kaggle_username.py")
    )["normalize_username"]
    assert normalize_username("muhammadhossen") == "muhammadhossen"
    assert normalize_username('"muhammadhossen"') == "muhammadhossen"
    assert normalize_username('"muhammadhossen",') == "muhammadhossen"
    assert normalize_username('"username": "muhammadhossen",') == "muhammadhossen"
    assert normalize_username('{"username": "muhammadhossen", "key": "redacted"}') == "muhammadhossen"


def test_kaggle_username_normalizer_rejects_display_names_without_echoing_input() -> None:
    import runpy
    import pytest

    normalize_username = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "kaggle_username.py")
    )["normalize_username"]
    with pytest.raises(ValueError, match="valid Kaggle profile handle"):
        normalize_username("Muhammad Hossen")


def test_kaggle_workflows_normalize_username_before_kaggle_kernel_or_dataset_ids() -> None:
    root = Path(__file__).resolve().parents[1]
    for relative in (".github/workflows/validation.yml", ".github/workflows/continuous-kaggle-training.yml"):
        source = (root / relative).read_text(encoding="utf-8")
        assert "python kaggle_username.py" in source
        assert 'export KAGGLE_USERNAME="$(cat .kaggle-username)"' in source
        assert 'export KAGGLE_STATE_DATASET="$KAGGLE_USERNAME/artemus-training-state"' in source


def test_gpu_notebook_reports_accelerator_diagnostics_before_failing_fast() -> None:
    notebook_path = Path(__file__).resolve().parents[1] / "notebooks" / "kaggle-gpu-training.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cell = next(cell for cell in notebook["cells"] if cell.get("id") == "gpu-check")
    source = "".join(cell.get("source", []))
    assert "torch.version.cuda" in source
    assert "torch.cuda.device_count()" in source
    assert "CUDA_VISIBLE_DEVICES" in source
    assert 'shutil.which("nvidia-smi")' in source
    assert "phone verification" in source

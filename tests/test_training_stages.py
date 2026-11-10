from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_f3_is_a_verified_f2_continuation_not_an_sft_checkpoint() -> None:
    config = json.loads((ROOT / "scripts/cpu/f3.json").read_text(encoding="utf-8"))
    assert config["parent"] == "ckpt/artemus-f2-edu2-6m.pt"
    assert config["parent_sha256"] == (
        "c4a791f9941ef4488c74a51bf2a46fce40b0764aa8bd81683c35385939efd584"
    )
    assert config["expected_parent_step"] == 17500
    assert config["expected_parent_response_only"] is False
    assert config["expected_response_only"] is False
    assert config["expected_phase"] == "continued_pretraining"
    assert config["steps"] == 4000
    assert "--allow-objective-change" not in config["args"]


def test_f3_uses_pinned_sources_and_replay_mixture() -> None:
    workflow = (ROOT / ".github/workflows/artemis-f3-training.yml").read_text(
        encoding="utf-8"
    )
    assert "workflow_run:" in workflow
    assert "ArtemisLM F2 to S3 continuation" in workflow
    assert "f87fe9a2ddf130c28295796ac89d19ec23256b24" in workflow
    assert "e92b25a616738fe95dc186b64dfb19f9c8525594" in workflow
    assert "FineWeb-Edu shard 008" in workflow
    assert "FineMath-4+ shard 00012" in workflow
    assert "0.85" in workflow and "0.15" in workflow

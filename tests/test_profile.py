import pytest

from artemis.config import ModelConfig
from artemis.profile import profile_model


def test_profile_avoids_double_counting_tied_embedding():
    config = ModelConfig(vocab_size=512, block_size=32, n_layer=2, n_head=4, n_embd=64)
    result = profile_model(config)
    assert result["model_parameters"] == sum(result["parameters_by_component"].values())
    assert result["parameters_by_component"]["position_embedding"] == 32 * 64


def test_gqa_reduces_theoretical_kv_cache_estimate():
    full = ModelConfig(vocab_size=512, block_size=128, n_layer=4, n_head=8, n_embd=128)
    grouped = ModelConfig(
        vocab_size=512, block_size=128, n_layer=4, n_head=8, n_embd=128,
        n_kv_head=2, use_rope=True,
    )
    full_report = profile_model(full, batch_size=2, context_length=64, dtype_bytes=2)
    gqa_report = profile_model(grouped, batch_size=2, context_length=64, dtype_bytes=2)
    assert gqa_report["kv_cache_estimate"]["gqa_cache_bytes"] == (
        full_report["kv_cache_estimate"]["full_mha_cache_bytes"] // 4
    )
    assert gqa_report["kv_cache_estimate"]["reduction_fraction_vs_full_mha"] == 0.75


def test_profile_validates_context_and_dtype():
    config = ModelConfig(vocab_size=512, block_size=32, n_layer=1, n_head=2, n_embd=16)
    with pytest.raises(ValueError, match="context_length"):
        profile_model(config, context_length=33)
    with pytest.raises(ValueError, match="dtype_bytes"):
        profile_model(config, dtype_bytes=0)

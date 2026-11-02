import torch
import pytest

from artemis.bench_kv_cache import benchmark_generation, cache_payload_bytes
from artemis.config import ModelConfig
from artemis.model import ArtemisLM


def test_cache_payload_byte_count_uses_actual_tensor_dtype():
    key = torch.zeros(1, 2, 5, 8, dtype=torch.float32)
    value = torch.zeros_like(key)
    assert cache_payload_bytes(((key, value),)) == 1 * 2 * 5 * 8 * 4 * 2


def test_benchmark_checks_cached_output_matches_uncached():
    torch.manual_seed(21)
    model = ArtemisLM(ModelConfig(
        vocab_size=257, block_size=8, n_layer=1, n_head=2, n_embd=16,
        n_kv_head=1, use_rope=True,
    ))
    prompt = torch.tensor([[1, 2, 3, 4, 5, 6]])
    result = benchmark_generation(
        model, prompt, new_tokens=4, warmup=0, repeats=2,
        device=torch.device("cpu"),
    )
    assert result["cached_and_uncached_outputs_match"] is True
    assert result["kv_cache_payload_bytes"] > 0
    assert result["cached_latency_ms"] > 0
    assert result["uncached_latency_ms"] > 0


def test_benchmark_validates_prompt_and_repeats():
    model = ArtemisLM(ModelConfig(block_size=8, n_layer=1, n_head=2, n_embd=16))
    prompt = torch.tensor([[1, 2]])
    with pytest.raises(ValueError, match="repeats"):
        benchmark_generation(model, prompt, new_tokens=1, repeats=0)
    with pytest.raises(ValueError, match="non-empty"):
        benchmark_generation(model, torch.empty((1, 0), dtype=torch.long), new_tokens=1)

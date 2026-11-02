import torch

from artemis.config import ModelConfig
from artemis.model import ArtemisLM


def test_output_projection_shares_embedding_weights_after_initialization() -> None:
    model = ArtemisLM(ModelConfig(block_size=16, n_layer=2, n_head=2, n_embd=16))
    assert model.output.weight is model.token_embedding.weight


def test_generation_appends_tokens_and_restores_training_mode() -> None:
    model = ArtemisLM(ModelConfig(block_size=8, n_layer=1, n_head=2, n_embd=16))
    model.train()
    prompt = torch.tensor([[65, 66, 67]], dtype=torch.long)
    generated = model.generate(prompt, max_new_tokens=5, temperature=1.0, top_k=10)
    assert generated.shape == (1, 8)
    assert model.training


def test_cached_generation_matches_uncached_reference_across_window_rollover():
    torch.manual_seed(101)
    model = ArtemisLM(
        ModelConfig(
            vocab_size=257, block_size=8, n_layer=2, n_head=2, n_embd=16,
            n_kv_head=1, use_rope=True,
        )
    ).eval()
    prompt = torch.tensor([[10, 11, 12, 13, 14, 15]], dtype=torch.long)
    with torch.inference_mode():
        cached = model.generate(
            prompt, 14, temperature=1.0, top_k=1, use_cache=True
        )
        uncached = model.generate(
            prompt, 14, temperature=1.0, top_k=1, use_cache=False
        )
    assert torch.equal(cached, uncached)


def test_cached_generation_restores_training_mode_and_handles_zero_tokens():
    model = ArtemisLM(ModelConfig(block_size=8, n_layer=1, n_head=2, n_embd=16))
    model.train()
    prompt = torch.tensor([[65, 66, 67]], dtype=torch.long)
    unchanged = model.generate(prompt, 0, use_cache=True)
    assert torch.equal(unchanged, prompt)
    generated = model.generate(prompt, 3, temperature=1.0, top_k=1, use_cache=True)
    assert generated.shape == (1, 6)
    assert model.training

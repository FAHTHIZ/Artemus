import torch
import pytest

from artemis.config import ModelConfig
from artemis.model import ArtemisLM, apply_rope


def small_model() -> ArtemisLM:
    return ArtemisLM(ModelConfig(block_size=16, n_layer=2, n_head=2, n_embd=16, dropout=0.0))


def test_forward_shape_and_loss() -> None:
    model = small_model()
    tokens = torch.randint(0, 257, (3, 12))
    logits, loss = model(tokens, tokens)
    assert logits.shape == (3, 12, 257)
    assert loss is not None and torch.isfinite(loss)


def test_attention_does_not_use_future_tokens() -> None:
    model = small_model().eval()
    first = torch.randint(0, 257, (1, 12))
    second = first.clone()
    second[:, 8:] = torch.randint(0, 257, (1, 4))
    with torch.inference_mode():
        logits_a, _ = model(first)
        logits_b, _ = model(second)
    assert torch.allclose(logits_a[:, :8], logits_b[:, :8], atol=1e-6, rtol=1e-5)


def test_rejects_sequence_longer_than_configured_context() -> None:
    model = small_model()
    tokens = torch.randint(0, 257, (1, 17))
    with pytest.raises(ValueError, match="exceeds block_size"):
        model(tokens)


def compact_model() -> ArtemisLM:
    return ArtemisLM(ModelConfig(
        vocab_size=512, block_size=32, n_layer=2, n_head=4, n_embd=64,
        n_kv_head=2, use_rope=True, use_qk_norm=True, use_swiglu=True, dropout=0.0,
    ))


def test_gqa_rope_qk_norm_swiglu_forward_and_loss_are_finite():
    model = compact_model()
    tokens = torch.randint(0, 512, (2, 24))
    logits, loss = model(tokens, tokens)
    assert logits.shape == (2, 24, 512)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_gqa_reduces_attention_projection_parameters():
    mha = ArtemisLM(ModelConfig(vocab_size=512, block_size=32, n_layer=1, n_head=4, n_embd=64))
    gqa = ArtemisLM(ModelConfig(
        vocab_size=512, block_size=32, n_layer=1, n_head=4, n_embd=64,
        n_kv_head=1, use_rope=True,
    ))
    mha_n = sum(p.numel() for p in mha.blocks[0].attention.parameters())
    gqa_n = sum(p.numel() for p in gqa.blocks[0].attention.parameters())
    assert gqa_n < mha_n
    assert gqa.position_embedding is None


def test_gqa_attention_preserves_causal_mask():
    torch.manual_seed(123)
    model = compact_model().eval()
    first = torch.randint(0, 512, (1, 20))
    changed = first.clone()
    changed[:, 12:] = torch.randint(0, 512, (1, 8))
    with torch.inference_mode():
        logits_a, _ = model(first)
        logits_b, _ = model(changed)
    assert torch.allclose(logits_a[:, :12], logits_b[:, :12], atol=1e-6, rtol=1e-5)


def test_rope_preserves_tensor_shape_and_norm():
    x = torch.randn(2, 4, 16, 8)
    rotated = apply_rope(x)
    assert rotated.shape == x.shape
    assert torch.allclose(x.norm(dim=-1), rotated.norm(dim=-1), atol=1e-5, rtol=1e-5)


def test_config_validates_gqa_head_divisibility():
    with pytest.raises(ValueError, match="divisible by n_kv_head"):
        ModelConfig(n_embd=64, n_head=4, n_kv_head=3)


def test_config_validates_rope_even_head_dimension():
    with pytest.raises(ValueError, match="even attention head dimension"):
        ModelConfig(n_embd=60, n_head=4, use_rope=True)


def test_legacy_config_dict_defaults_to_original_architecture():
    config = ModelConfig.from_dict({
        "vocab_size": 257, "block_size": 16, "n_layer": 2,
        "n_head": 2, "n_embd": 16, "dropout": 0.0,
    })
    assert config.n_kv_head is None
    assert not config.use_rope and not config.use_qk_norm and not config.use_swiglu
    model = ArtemisLM(config)
    assert model.position_embedding is not None
    assert hasattr(model.blocks[0].attention, "qkv")
    assert hasattr(model.blocks[0].feed_forward, "net")


def test_factorized_token_embeddings_reduce_parameters_and_preserve_tying():
    base_config = ModelConfig(
        vocab_size=4096, block_size=64, n_layer=2, n_head=4, n_embd=64
    )
    factor_config = ModelConfig(
        vocab_size=4096, block_size=64, n_layer=2, n_head=4, n_embd=64,
        embedding_dim=32,
    )
    base = ArtemisLM(base_config)
    factor = ArtemisLM(factor_config)
    assert factor.token_embedding.weight is factor.output.weight
    assert factor.token_embedding.embedding_dim == 32
    assert factor.input_projection.weight.shape == (64, 32)
    assert factor.output_projection.weight.shape == (32, 64)
    assert sum(p.numel() for p in factor.parameters()) < sum(
        p.numel() for p in base.parameters()
    )


def test_factorized_embedding_forward_backward_is_finite():
    config = ModelConfig(
        vocab_size=512, block_size=32, n_layer=2, n_head=4, n_embd=64,
        embedding_dim=32, n_kv_head=2, use_rope=True, use_swiglu=True,
    )
    model = ArtemisLM(config)
    tokens = torch.randint(0, 512, (2, 24))
    logits, loss = model(tokens, tokens)
    assert logits.shape == (2, 24, 512)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_factorized_embedding_old_config_keeps_legacy_weight_shapes():
    config = ModelConfig.from_dict({
        "vocab_size": 257, "block_size": 16, "n_layer": 2,
        "n_head": 2, "n_embd": 16, "dropout": 0.0,
    })
    model = ArtemisLM(config)
    assert model.token_embedding.weight.shape == (257, 16)
    assert model.output.weight is model.token_embedding.weight
    assert isinstance(model.input_projection, torch.nn.Identity)
    assert isinstance(model.output_projection, torch.nn.Identity)


@pytest.mark.parametrize(
    "config",
    [
        ModelConfig(vocab_size=512, block_size=24, n_layer=2, n_head=4, n_embd=64),
        ModelConfig(
            vocab_size=512, block_size=24, n_layer=2, n_head=4, n_embd=64,
            n_kv_head=2, use_rope=True, use_qk_norm=True, use_swiglu=True,
            embedding_dim=32,
        ),
    ],
)
def test_token_by_token_kv_cache_matches_full_forward(config):
    torch.manual_seed(17)
    model = ArtemisLM(config).eval()
    tokens = torch.randint(0, config.vocab_size, (2, 11))
    with torch.inference_mode():
        expected, _ = model(tokens)
        cache = None
        chunks = []
        for position in range(tokens.size(1)):
            logits, cache = model.forward_with_cache(
                tokens[:, position:position + 1],
                cache,
                position_offset=position,
            )
            chunks.append(logits)
        actual = torch.cat(chunks, dim=1)
    assert torch.allclose(actual, expected, atol=2e-5, rtol=2e-5)


def test_multi_token_cached_chunk_uses_correct_non_square_causal_mask():
    torch.manual_seed(23)
    model = compact_model().eval()
    tokens = torch.randint(0, 512, (2, 13))
    with torch.inference_mode():
        expected, _ = model(tokens)
        first_logits, cache = model.forward_with_cache(tokens[:, :5])
        second_logits, cache = model.forward_with_cache(
            tokens[:, 5:10], cache, position_offset=5
        )
        third_logits, cache = model.forward_with_cache(
            tokens[:, 10:], cache, position_offset=10
        )
        actual = torch.cat((first_logits, second_logits, third_logits), dim=1)
    assert torch.allclose(actual, expected, atol=2e-5, rtol=2e-5)
    assert cache[0][0].shape == (2, 2, 13, 16)
    assert cache[0][1].shape == (2, 2, 13, 16)


def test_gqa_cache_stores_kv_heads_not_expanded_query_heads():
    model = compact_model().eval()
    tokens = torch.randint(0, 512, (1, 7))
    with torch.inference_mode():
        _, cache = model.forward_with_cache(tokens)
    for key, value in cache:
        assert key.shape[1] == model.config.n_kv_head
        assert value.shape[1] == model.config.n_kv_head


def test_cached_forward_rejects_wrong_position_offset():
    model = compact_model().eval()
    tokens = torch.randint(0, 512, (1, 5))
    with torch.inference_mode():
        _, cache = model.forward_with_cache(tokens)
        with pytest.raises(ValueError, match="position_offset must equal"):
            model.forward_with_cache(tokens[:, :1], cache, position_offset=4)

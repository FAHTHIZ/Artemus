import torch

from artemis.config import ModelConfig
from artemis.model import ArtemisLM


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
    try:
        model(tokens)
    except ValueError as exc:
        assert "exceeds block_size" in str(exc)
    else:
        raise AssertionError("expected sequence-length validation to fail")

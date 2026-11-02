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

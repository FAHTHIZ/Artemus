import torch

from artemis.config import ModelConfig
from artemis.model import ArtemisLM


def test_historical_state_dict_loads_with_legacy_config_defaults():
    legacy_values = {
        "vocab_size": 257,
        "block_size": 32,
        "n_layer": 2,
        "n_head": 2,
        "n_embd": 16,
        "dropout": 0.0,
    }
    legacy_config = ModelConfig.from_dict(legacy_values)
    original = ArtemisLM(legacy_config)
    checkpoint_state = original.state_dict()

    reloaded = ArtemisLM(ModelConfig.from_dict(legacy_values))
    result = reloaded.load_state_dict(checkpoint_state, strict=True)
    assert not result.missing_keys
    assert not result.unexpected_keys

    tokens = torch.randint(0, legacy_config.vocab_size, (2, 12))
    with torch.inference_mode():
        expected, _ = original.eval()(tokens)
        actual, _ = reloaded.eval()(tokens)
    assert torch.equal(expected, actual)


def test_new_config_round_trips_all_architecture_options():
    config = ModelConfig(
        vocab_size=512,
        block_size=128,
        n_layer=3,
        n_head=8,
        n_embd=128,
        n_kv_head=2,
        use_rope=True,
        use_qk_norm=True,
        use_swiglu=True,
        rope_theta=500000.0,
    )
    restored = ModelConfig.from_dict(config.to_dict())
    assert restored == config

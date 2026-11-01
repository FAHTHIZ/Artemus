import torch

from artemis.train import get_batch


def test_get_batch_returns_shifted_sequences() -> None:
    data = torch.arange(100, dtype=torch.long)
    x, y = get_batch(data, batch_size=4, block_size=8, device=torch.device("cpu"))
    assert x.shape == (4, 8)
    assert y.shape == (4, 8)
    assert torch.equal(y[:, :-1], x[:, 1:])

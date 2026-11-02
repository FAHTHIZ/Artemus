import torch
import pytest

from artemis.train import build_response_loss_mask, get_batch
from artemis.tokenizer import SentencePieceTokenizer


def test_get_batch_returns_shifted_sequences() -> None:
    data = torch.arange(100, dtype=torch.long)
    x, y = get_batch(data, batch_size=4, block_size=8, device=torch.device("cpu"))
    assert x.shape == (4, 8)
    assert y.shape == (4, 8)
    assert torch.equal(y[:, :-1], x[:, 1:])


def test_get_batch_masks_non_assistant_targets():
    data = torch.arange(100, dtype=torch.long)
    target_mask = torch.zeros(100, dtype=torch.bool)
    target_mask[20:35] = True
    x, y = get_batch(
        data, batch_size=4, block_size=8, device=torch.device("cpu"),
        target_mask=target_mask,
    )
    assert y.shape == (4, 8)
    assert torch.all((y == -100) | (y == torch.arange(1, 9)[None, :] + x[:, :1]))


def test_get_batch_rejects_mask_with_no_assistant_targets():
    data = torch.arange(100, dtype=torch.long)
    with pytest.raises(RuntimeError, match="assistant response targets"):
        get_batch(
            data, batch_size=2, block_size=8, device=torch.device("cpu"),
            target_mask=torch.zeros(100, dtype=torch.bool),
        )


def test_response_loss_mask_marks_only_assistant_text_and_end(tmp_path):
    corpus = tmp_path / "training.txt"
    corpus.write_text(
        ("<|prompter|> Explain a loop.\n<|assistant|> A loop repeats steps.\n"
         "<|end|>\n") * 40,
        encoding="utf-8",
    )
    from artemis.tokenizer import SentencePieceTokenizer
    tokenizer = SentencePieceTokenizer.train(corpus, tmp_path / "mask-tokenizer", vocab_size=512)
    ids = torch.tensor(
        tokenizer.encode(
            "<|prompter|> Question text <|assistant|> Answer text <|end|>"
        ),
        dtype=torch.long,
    )
    mask = build_response_loss_mask(ids, tokenizer)
    pieces = [tokenizer.processor.id_to_piece(int(token_id)) for token_id in ids]
    assistant = pieces.index("<|assistant|>")
    end = pieces.index("<|end|>")
    assert not mask[: assistant + 1].any()
    assert mask[assistant + 1 : end].all()
    assert mask[end]
    assert not mask[end + 1 :].any()

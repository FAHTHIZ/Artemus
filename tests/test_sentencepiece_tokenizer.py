from pathlib import Path

from artemis.tokenizer import ByteTokenizer, SentencePieceTokenizer


def test_byte_tokenizer_remains_backward_compatible():
    tokenizer = ByteTokenizer()
    text = "Hello, world!"
    assert tokenizer.decode(tokenizer.encode(text)) == text
    assert tokenizer.encode("A", add_eos=True)[-1] == tokenizer.eos_token_id


def test_sentencepiece_round_trips_unicode_and_chat_markers(tmp_path: Path):
    corpus = tmp_path / "train.txt"
    corpus.write_text(
        ("<|prompter|> What is a variable?\n"
         "<|assistant|> A variable stores a value.\n"
         "<|end|>\n"
         "A second English tokenizer sample. English and code: def add(a, b): return a + b\n") * 80,
        encoding="utf-8",
    )
    tokenizer = SentencePieceTokenizer.train(
        corpus, tmp_path / "tiny-tokenizer", vocab_size=512
    )
    text = "<|prompter|> English test <|assistant|> A variable stores a value."
    ids = tokenizer.encode(text)
    assert tokenizer.decode(ids) == text
    assert tokenizer.encode("<|prompter|>") == [tokenizer.processor.piece_to_id("<|prompter|>")]
    assert tokenizer.encode("<|assistant|>") == [tokenizer.processor.piece_to_id("<|assistant|>")]
    assert tokenizer.stop_token_id == tokenizer.processor.piece_to_id("<|end|>")
    assert tokenizer.encode("hello", add_eos=True)[-1] == tokenizer.eos_token_id


def test_sentencepiece_tokenizer_can_be_restored_from_serialized_proto(tmp_path: Path):
    corpus = tmp_path / "train.txt"
    corpus.write_text(("A small tokenizer test. an English tokenizer test.\n") * 100, encoding="utf-8")
    tokenizer = SentencePieceTokenizer.train(corpus, tmp_path / "serialized", vocab_size=512)
    restored = SentencePieceTokenizer.from_serialized_model(tokenizer.serialized_model)
    sample = "Hello, world!"
    assert restored.decode(restored.encode(sample)) == sample
    assert restored.vocab_size == tokenizer.vocab_size

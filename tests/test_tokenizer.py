from artemis.tokenizer import ByteTokenizer, EOS_TOKEN_ID, VOCAB_SIZE


def test_utf8_round_trip_for_english() -> None:
    tokenizer = ByteTokenizer()
    text = "Artemis: coding, reasoning, English, and Python."
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_eos_token_is_appended_and_skipped_on_decode() -> None:
    tokenizer = ByteTokenizer()
    ids = tokenizer.encode("hello", add_eos=True)
    assert ids[-1] == EOS_TOKEN_ID
    assert tokenizer.decode(ids) == "hello"
    assert VOCAB_SIZE == 257

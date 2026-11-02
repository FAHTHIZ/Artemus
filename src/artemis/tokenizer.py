"""A deterministic UTF-8 byte tokenizer for pipeline validation.

This tokenizer is intentionally simple and is not proposed as the final
production tokenizer. It has no learned vocabulary and round-trips valid UTF-8.
"""

from collections.abc import Iterable

EOS_TOKEN_ID = 256
VOCAB_SIZE = 257


class ByteTokenizer:
    """Map UTF-8 bytes to token IDs 0..255 and reserve 256 for EOS."""

    vocab_size = VOCAB_SIZE
    eos_token_id = EOS_TOKEN_ID

    def encode(self, text: str, *, add_eos: bool = False) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        token_ids = list(text.encode("utf-8"))
        if add_eos:
            token_ids.append(self.eos_token_id)
        return token_ids

    def decode(self, token_ids: Iterable[int]) -> str:
        byte_values = bytearray()
        for token_id in token_ids:
            value = int(token_id)
            if value == self.eos_token_id:
                continue
            if not 0 <= value <= 255:
                raise ValueError(f"invalid byte-token ID: {value}")
            byte_values.append(value)
        return byte_values.decode("utf-8", errors="replace")

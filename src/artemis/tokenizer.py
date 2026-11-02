"""Tokenizers used by Artemis experiments.

The byte tokenizer remains supported for old checkpoints. New conversation
experiments should use SentencePiece so ordinary words need fewer tokens and
chat markers remain dedicated symbols.
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import sentencepiece as spm

EOS_TOKEN_ID = 256
VOCAB_SIZE = 257
CHAT_SPECIAL_TOKENS = ["<|prompter|>", "<|assistant|>", "<|end|>"]


class ByteTokenizer:
    """Legacy UTF-8 byte tokenizer; IDs 0..255 plus EOS=256."""

    vocab_size = VOCAB_SIZE
    eos_token_id = EOS_TOKEN_ID
    stop_token_id = EOS_TOKEN_ID
    name = "utf8-byte-v1"

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


class SentencePieceTokenizer:
    """SentencePiece BPE tokenizer with byte fallback and dedicated chat markers."""

    name = "sentencepiece-bpe-v1"

    def __init__(
        self,
        *,
        model_file: Path | None = None,
        model_proto: bytes | None = None,
    ) -> None:
        if (model_file is None) == (model_proto is None):
            raise ValueError("provide exactly one of model_file or model_proto")
        self.processor = spm.SentencePieceProcessor()
        if model_file is not None:
            loaded = self.processor.Load(str(model_file))
        else:
            loaded = self.processor.LoadFromSerializedProto(model_proto)
        if not loaded:
            raise ValueError("could not load SentencePiece model")
        self.vocab_size = int(self.processor.get_piece_size())
        self.eos_token_id = int(self.processor.eos_id())
        self.stop_token_id = int(self.processor.piece_to_id("<|end|>"))
        if self.vocab_size < 300 or self.eos_token_id < 0 or self.stop_token_id < 0:
            raise ValueError("SentencePiece model is missing expected vocabulary/EOS")

    @classmethod
    def train(
        cls,
        input_path: Path,
        model_prefix: Path,
        *,
        vocab_size: int = 2048,
    ) -> "SentencePieceTokenizer":
        if not input_path.is_file():
            raise FileNotFoundError(f"Tokenizer training data does not exist: {input_path}")
        if vocab_size < 512:
            raise ValueError("SentencePiece vocab_size must be at least 512 with byte fallback")
        model_prefix.parent.mkdir(parents=True, exist_ok=True)
        spm.SentencePieceTrainer.train(
            input=str(input_path),
            model_prefix=str(model_prefix),
            model_type="bpe",
            vocab_size=vocab_size,
            character_coverage=0.9995,
            byte_fallback=True,
            pad_id=0,
            unk_id=1,
            bos_id=-1,
            eos_id=2,
            user_defined_symbols=CHAT_SPECIAL_TOKENS,
            hard_vocab_limit=False,
            num_threads=2,
            add_dummy_prefix=False,
        )
        return cls(model_file=model_prefix.with_suffix(".model"))

    @classmethod
    def from_serialized_model(cls, model_proto: bytes) -> "SentencePieceTokenizer":
        return cls(model_proto=model_proto)

    @property
    def serialized_model(self) -> bytes:
        return self.processor.serialized_model_proto()

    def encode(self, text: str, *, add_eos: bool = False) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        token_ids = list(self.processor.encode(text, out_type=int))
        if add_eos:
            token_ids.append(self.eos_token_id)
        return token_ids

    def decode(self, token_ids: Iterable[int]) -> str:
        clean_ids = [int(token_id) for token_id in token_ids if int(token_id) != self.eos_token_id]
        return self.processor.decode(clean_ids)

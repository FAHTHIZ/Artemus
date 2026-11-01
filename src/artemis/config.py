"""Configuration for the initial decoder-only Transformer baseline."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 257  # 256 UTF-8 byte values plus one end-of-document token
    block_size: int = 128
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.vocab_size < 257:
            raise ValueError("vocab_size must be at least 257 for the byte tokenizer")
        if self.block_size < 2:
            raise ValueError("block_size must be at least 2")
        if self.n_layer < 1 or self.n_head < 1 or self.n_embd < 1:
            raise ValueError("n_layer, n_head, and n_embd must be positive")
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, int | float]) -> "ModelConfig":
        return cls(**values)

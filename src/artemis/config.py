"""Configuration for Artemis decoder-only Transformer experiments."""
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 257
    block_size: int = 128
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    # Defaults preserve the original model and old checkpoint structure.
    n_kv_head: int | None = None
    use_rope: bool = False
    use_qk_norm: bool = False
    use_swiglu: bool = False
    rope_theta: float = 10000.0

    def __post_init__(self) -> None:
        if self.vocab_size < 257:
            raise ValueError("vocab_size must be at least 257 for byte fallback")
        if self.block_size < 2:
            raise ValueError("block_size must be at least 2")
        if self.n_layer < 1 or self.n_head < 1 or self.n_embd < 1:
            raise ValueError("n_layer, n_head, and n_embd must be positive")
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        kv_heads = self.n_head if self.n_kv_head is None else self.n_kv_head
        if kv_heads < 1 or kv_heads > self.n_head:
            raise ValueError("n_kv_head must be between 1 and n_head")
        if self.n_head % kv_heads != 0:
            raise ValueError("n_head must be divisible by n_kv_head")
        if self.use_rope and (self.n_embd // self.n_head) % 2 != 0:
            raise ValueError("RoPE requires an even attention head dimension")
        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ModelConfig":
        # Missing keys use defaults so older checkpoints load unchanged.
        return cls(**values)

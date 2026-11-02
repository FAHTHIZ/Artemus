"""Decoder-only Transformer with opt-in efficient attention/FFN variants."""
import torch
from torch import nn
from torch.nn import functional as F

from artemis.config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * scale.to(dtype=x.dtype) * self.weight


def apply_rope(x: torch.Tensor, theta: float = 10000.0) -> torch.Tensor:
    """Apply rotary position embeddings to [batch, heads, time, head_dim]."""
    if x.ndim != 4:
        raise ValueError("RoPE expects [batch, heads, time, head_dim]")
    head_dim = x.size(-1)
    if head_dim % 2:
        raise ValueError("RoPE requires an even head dimension")
    seq_len = x.size(-2)
    positions = torch.arange(seq_len, device=x.device, dtype=torch.float32)
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=x.device, dtype=torch.float32) / head_dim)
    )
    angles = torch.outer(positions, inv_freq)
    cos = angles.cos().to(dtype=x.dtype)[None, None, :, :]
    sin = angles.sin().to(dtype=x.dtype)[None, None, :, :]
    even, odd = x[..., 0::2], x[..., 1::2]
    rotated_even = even * cos - odd * sin
    rotated_odd = even * sin + odd * cos
    return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_head if config.n_kv_head is None else config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.use_rope = config.use_rope
        self.rope_theta = config.rope_theta
        self.use_qk_norm = config.use_qk_norm

        if self.n_kv_head == self.n_head:
            # Preserve old MHA state-dict keys and tensor shapes.
            self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        else:
            self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
            self.kv_proj = nn.Linear(
                config.n_embd, 2 * self.n_kv_head * self.head_dim, bias=False
            )
        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.resid_dropout = nn.Dropout(config.dropout)

    def _split_q(self, t: torch.Tensor, batch: int, seq: int) -> torch.Tensor:
        return t.view(batch, seq, self.n_head, self.head_dim).transpose(1, 2)

    def _split_kv(self, t: torch.Tensor, batch: int, seq: int) -> torch.Tensor:
        return t.view(batch, seq, self.n_kv_head, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, width = x.shape
        if self.n_kv_head == self.n_head:
            q_raw, k_raw, v_raw = self.qkv(x).chunk(3, dim=-1)
        else:
            q_raw = self.q_proj(x)
            k_raw, v_raw = self.kv_proj(x).chunk(2, dim=-1)

        q = self._split_q(q_raw, batch, seq_len)
        k = self._split_kv(k_raw, batch, seq_len)
        v = self._split_kv(v_raw, batch, seq_len)

        if self.use_qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        if self.use_rope:
            q, k = apply_rope(q, self.rope_theta), apply_rope(k, self.rope_theta)
        if self.n_kv_head != self.n_head:
            groups = self.n_head // self.n_kv_head
            k = k.repeat_interleave(groups, dim=1)
            v = v.repeat_interleave(groups, dim=1)

        attended = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, seq_len, width)
        return self.resid_dropout(self.proj(attended))


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.use_swiglu = config.use_swiglu
        if self.use_swiglu:
            hidden = ((8 * config.n_embd // 3 + 7) // 8) * 8
            self.gate_proj = nn.Linear(config.n_embd, hidden, bias=False)
            self.up_proj = nn.Linear(config.n_embd, hidden, bias=False)
            self.down_proj = nn.Linear(hidden, config.n_embd, bias=False)
            self.dropout = nn.Dropout(config.dropout)
        else:
            # Preserve old GELU feed-forward parameter keys for old checkpoints.
            self.net = nn.Sequential(
                nn.Linear(config.n_embd, 4 * config.n_embd),
                nn.GELU(),
                nn.Linear(4 * config.n_embd, config.n_embd),
                nn.Dropout(config.dropout),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_swiglu:
            gated = F.silu(self.gate_proj(x)) * self.up_proj(x)
            return self.dropout(self.down_proj(gated))
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(config.n_embd)
        self.attention = CausalSelfAttention(config)
        self.norm2 = RMSNorm(config.n_embd)
        self.feed_forward = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        return x + self.feed_forward(self.norm2(x))


class ArtemisLM(nn.Module):
    """A compact GPT-style language model initialized from random weights."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding_dim = (
            config.n_embd if config.embedding_dim is None else config.embedding_dim
        )
        self.token_embedding = nn.Embedding(config.vocab_size, self.embedding_dim)
        self.input_projection = (
            nn.Identity()
            if self.embedding_dim == config.n_embd
            else nn.Linear(self.embedding_dim, config.n_embd, bias=False)
        )
        self.output_projection = (
            nn.Identity()
            if self.embedding_dim == config.n_embd
            else nn.Linear(config.n_embd, self.embedding_dim, bias=False)
        )
        self.position_embedding = (
            None if config.use_rope else nn.Embedding(config.block_size, config.n_embd)
        )
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(*(TransformerBlock(config) for _ in range(config.n_layer)))
        self.final_norm = RMSNorm(config.n_embd)
        self.output = nn.Linear(self.embedding_dim, config.vocab_size, bias=False)
        self.apply(self._initialise)
        self.output.weight = self.token_embedding.weight

    @staticmethod
    def _initialise(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self, token_ids: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence]")
        _, seq_len = token_ids.shape
        if seq_len > self.config.block_size:
            raise ValueError(
                f"sequence length {seq_len} exceeds block_size {self.config.block_size}"
            )
        x = self.input_projection(self.token_embedding(token_ids))
        if self.position_embedding is not None:
            positions = torch.arange(seq_len, device=token_ids.device)
            x = x + self.position_embedding(positions)[None, :, :]
        x = self.embedding_dropout(x)
        x = self.blocks(x)
        logits = self.output(self.output_projection(self.final_norm(x)))
        loss = None
        if targets is not None:
            if targets.shape != token_ids.shape:
                raise ValueError("targets must have the same shape as token_ids")
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )
        return logits, loss

    @torch.inference_mode()
    def generate(
        self,
        token_ids: torch.Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        if token_ids.ndim != 2 or token_ids.size(1) == 0:
            raise ValueError("token_ids must be a non-empty [batch, sequence] tensor")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero")
        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be positive when provided")
        was_training = self.training
        self.eval()
        generated = token_ids
        try:
            for _ in range(max_new_tokens):
                context = generated[:, -self.config.block_size :]
                logits, _ = self(context)
                next_logits = logits[:, -1, :] / temperature
                if top_k is not None:
                    k = min(top_k, next_logits.size(-1))
                    values, _ = torch.topk(next_logits, k)
                    next_logits[next_logits < values[:, [-1]]] = float("-inf")
                probabilities = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)
                generated = torch.cat((generated, next_token), dim=1)
                if eos_token_id is not None and torch.all(next_token == eos_token_id):
                    break
        finally:
            self.train(was_training)
        return generated

"""Decoder-only Transformer with compact architecture options and incremental KV caching."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from artemis.config import ModelConfig

# One per layer: (key, value), each [batch, kv_heads, cached_tokens, head_dim].
LayerKVCache = tuple[torch.Tensor, torch.Tensor]
KVCache = tuple[LayerKVCache, ...]


@dataclass
class StaticKVCache:
    """Preallocated, per-layer K/V buffers for inference-only decoding."""

    keys: tuple[torch.Tensor, ...]
    values: tuple[torch.Tensor, ...]
    length: int = 0

    @property
    def capacity(self) -> int:
        if not self.keys:
            return 0
        return self.keys[0].size(2)

    def reset(self) -> None:
        """Reuse the allocated buffers from position zero without reallocating."""
        self.length = 0


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * scale.to(dtype=x.dtype) * self.weight


def apply_rope(
    x: torch.Tensor,
    theta: float = 10000.0,
    *,
    position_offset: int = 0,
) -> torch.Tensor:
    """Apply RoPE to [batch, heads, time, head_dim] at absolute positions."""
    if x.ndim != 4:
        raise ValueError("RoPE expects [batch, heads, time, head_dim]")
    if position_offset < 0:
        raise ValueError("position_offset must be non-negative")
    head_dim = x.size(-1)
    if head_dim % 2:
        raise ValueError("RoPE requires an even head dimension")
    seq_len = x.size(-2)
    positions = torch.arange(
        position_offset, position_offset + seq_len,
        device=x.device, dtype=torch.float32,
    )
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

    def _project_qkv(
        self,
        x: torch.Tensor,
        *,
        position_offset: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = x.shape
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
            q = apply_rope(q, self.rope_theta, position_offset=position_offset)
            k = apply_rope(k, self.rope_theta, position_offset=position_offset)
        return q, k, v

    def _expand_kv(
        self, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.n_kv_head == self.n_head:
            return k, v
        groups = self.n_head // self.n_kv_head
        return (
            k.repeat_interleave(groups, dim=1),
            v.repeat_interleave(groups, dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, seq_len, width = x.shape
        q, k, v = self._project_qkv(x, position_offset=0)
        k, v = self._expand_kv(k, v)
        attended = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).contiguous().view(x.size(0), seq_len, width)
        return self.resid_dropout(self.proj(attended))

    def forward_with_cache(
        self,
        x: torch.Tensor,
        past_key_value: LayerKVCache | None = None,
        *,
        position_offset: int = 0,
    ) -> tuple[torch.Tensor, LayerKVCache]:
        """Run attention and return a compact, unexpanded K/V cache.

        When a cache is supplied, cached keys are all from the past, so the
        attention mask is aligned to absolute query positions. This matters
        for multi-token cached chunks: SDPA's non-square is_causal=True mask is
        upper-left aligned and would mask some valid past keys incorrectly.
        """
        batch, seq_len, width = x.shape
        q, new_k, new_v = self._project_qkv(x, position_offset=position_offset)

        if past_key_value is None:
            past_len = 0
            k_all, v_all = new_k, new_v
        else:
            past_k, past_v = past_key_value
            if past_k.ndim != 4 or past_v.ndim != 4:
                raise ValueError("cached keys and values must be rank-4 tensors")
            if past_k.shape != past_v.shape:
                raise ValueError("cached keys and values must have identical shapes")
            if (
                past_k.size(0) != batch
                or past_k.size(1) != self.n_kv_head
                or past_k.size(3) != self.head_dim
            ):
                raise ValueError("past key/value cache shape is incompatible with this layer")
            if past_k.device != new_k.device or past_k.dtype != new_k.dtype:
                raise ValueError("past key/value cache device and dtype must match current inputs")
            past_len = past_k.size(2)
            if position_offset != past_len:
                raise ValueError(
                    "position_offset must equal the cached sequence length"
                )
            k_all = torch.cat((past_k, new_k), dim=2)
            v_all = torch.cat((past_v, new_v), dim=2)

        k_attn, v_attn = self._expand_kv(k_all, v_all)
        if past_len == 0:
            # Standard causal prefill; each prompt token sees only prior tokens.
            attended = F.scaled_dot_product_attention(
                q, k_attn, v_attn,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            # Query i corresponds to absolute position past_len + i. Allow keys
            # through that position, including current-chunk causal predecessors.
            query_positions = past_len + torch.arange(seq_len, device=x.device)
            key_positions = torch.arange(past_len + seq_len, device=x.device)
            allowed = key_positions[None, :] <= query_positions[:, None]
            attended = F.scaled_dot_product_attention(
                q, k_attn, v_attn,
                attn_mask=allowed,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )

        attended = attended.transpose(1, 2).contiguous().view(batch, seq_len, width)
        output = self.resid_dropout(self.proj(attended))
        # Cache stays at n_kv_head; never store the expanded query-head copy.
        return output, (k_all, v_all)

    def forward_with_static_cache(
        self,
        x: torch.Tensor,
        key_buffer: torch.Tensor,
        value_buffer: torch.Tensor,
        *,
        cache_position: int,
    ) -> torch.Tensor:
        """Write new K/V into preallocated buffers and attend over the active prefix."""
        batch, seq_len, width = x.shape
        if cache_position < 0:
            raise ValueError("cache_position must be non-negative")
        end_position = cache_position + seq_len
        expected_prefix = (batch, self.n_kv_head)
        if (
            key_buffer.ndim != 4
            or value_buffer.ndim != 4
            or key_buffer.shape != value_buffer.shape
            or key_buffer.shape[:2] != expected_prefix
            or key_buffer.size(3) != self.head_dim
        ):
            raise ValueError("static cache buffer shape is incompatible with this layer")
        if end_position > key_buffer.size(2):
            raise ValueError("static KV cache capacity exceeded")
        q, new_k, new_v = self._project_qkv(x, position_offset=cache_position)
        if (
            key_buffer.device != new_k.device
            or value_buffer.device != new_v.device
            or key_buffer.dtype != new_k.dtype
            or value_buffer.dtype != new_v.dtype
        ):
            raise ValueError("static KV cache device and dtype must match current inputs")

        # Inference-only API: copy this step's compact K/V into fixed buffer slots.
        key_buffer[:, :, cache_position:end_position, :].copy_(new_k)
        value_buffer[:, :, cache_position:end_position, :].copy_(new_v)
        k_all = key_buffer[:, :, :end_position, :]
        v_all = value_buffer[:, :, :end_position, :]
        k_attn, v_attn = self._expand_kv(k_all, v_all)

        if cache_position == 0:
            attended = F.scaled_dot_product_attention(
                q, k_attn, v_attn,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            query_positions = cache_position + torch.arange(seq_len, device=x.device)
            key_positions = torch.arange(end_position, device=x.device)
            allowed = key_positions[None, :] <= query_positions[:, None]
            attended = F.scaled_dot_product_attention(
                q, k_attn, v_attn,
                attn_mask=allowed,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
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

    def forward_with_cache(
        self,
        x: torch.Tensor,
        past_key_value: LayerKVCache | None = None,
        *,
        position_offset: int = 0,
    ) -> tuple[torch.Tensor, LayerKVCache]:
        attended, present = self.attention.forward_with_cache(
            self.norm1(x),
            past_key_value,
            position_offset=position_offset,
        )
        x = x + attended
        return x + self.feed_forward(self.norm2(x)), present

    def forward_with_static_cache(
        self,
        x: torch.Tensor,
        key_buffer: torch.Tensor,
        value_buffer: torch.Tensor,
        *,
        cache_position: int,
    ) -> torch.Tensor:
        attended = self.attention.forward_with_static_cache(
            self.norm1(x),
            key_buffer,
            value_buffer,
            cache_position=cache_position,
        )
        x = x + attended
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

    def forward_with_cache(
        self,
        token_ids: torch.Tensor,
        past_key_values: KVCache | None = None,
        *,
        position_offset: int | None = None,
    ) -> tuple[torch.Tensor, KVCache]:
        """Run a decoder prefill or cached continuation.

        The returned per-layer key/value tensors remain compact at n_kv_head.
        For a continuation, position_offset defaults to the cached context
        length and must agree with every layer's cache length.
        """
        if token_ids.ndim != 2 or token_ids.size(1) == 0:
            raise ValueError("token_ids must be a non-empty [batch, sequence] tensor")
        batch, seq_len = token_ids.shape
        if past_key_values is None:
            past_len = 0
        else:
            if len(past_key_values) != self.config.n_layer:
                raise ValueError("past_key_values must contain one cache per layer")
            first_key, first_value = past_key_values[0]
            if first_key.ndim != 4 or first_value.ndim != 4:
                raise ValueError("cached keys and values must be rank-4 tensors")
            past_len = first_key.size(2)
            if any(key.size(2) != past_len for key, _ in past_key_values):
                raise ValueError("all layer caches must have the same sequence length")

        if position_offset is None:
            position_offset = past_len
        if position_offset != past_len:
            raise ValueError("position_offset must equal the cached sequence length")
        if past_len + seq_len > self.config.block_size:
            raise ValueError(
                f"cached sequence length {past_len + seq_len} exceeds block_size "
                f"{self.config.block_size}; trim/re-prefill the context"
            )

        x = self.input_projection(self.token_embedding(token_ids))
        if self.position_embedding is not None:
            positions = torch.arange(
                position_offset, position_offset + seq_len, device=token_ids.device
            )
            x = x + self.position_embedding(positions)[None, :, :]
        x = self.embedding_dropout(x)

        present: list[LayerKVCache] = []
        for index, block in enumerate(self.blocks):
            layer_past = None if past_key_values is None else past_key_values[index]
            x, layer_present = block.forward_with_cache(
                x, layer_past, position_offset=position_offset
            )
            present.append(layer_present)

        logits = self.output(self.output_projection(self.final_norm(x)))
        return logits, tuple(present)

    @torch.inference_mode()
    def allocate_static_kv_cache(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> StaticKVCache:
        """Allocate fixed-capacity compact K/V buffers for inference."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        reference = next(self.parameters())
        selected_device = reference.device if device is None else device
        selected_dtype = reference.dtype if dtype is None else dtype
        keys: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        kv_heads = self.config.n_head if self.config.n_kv_head is None else self.config.n_kv_head
        head_dim = self.config.n_embd // self.config.n_head
        shape = (batch_size, kv_heads, self.config.block_size, head_dim)
        for _ in range(self.config.n_layer):
            keys.append(torch.empty(shape, device=selected_device, dtype=selected_dtype))
            values.append(torch.empty(shape, device=selected_device, dtype=selected_dtype))
        return StaticKVCache(tuple(keys), tuple(values))

    @torch.inference_mode()
    def forward_with_static_cache(
        self,
        token_ids: torch.Tensor,
        cache: StaticKVCache,
    ) -> tuple[torch.Tensor, StaticKVCache]:
        """Run inference using preallocated K/V buffers; no per-step K/V concatenation."""
        if self.training:
            raise RuntimeError("static KV cache is inference-only; call model.eval() first")
        if token_ids.ndim != 2 or token_ids.size(1) == 0:
            raise ValueError("token_ids must be a non-empty [batch, sequence] tensor")
        batch, seq_len = token_ids.shape
        end_position = cache.length + seq_len
        if len(cache.keys) != self.config.n_layer or len(cache.values) != self.config.n_layer:
            raise ValueError("static cache must contain one key and value buffer per layer")
        if cache.length < 0 or end_position > self.config.block_size:
            raise ValueError("static KV cache capacity exceeded")
        if cache.capacity != self.config.block_size:
            raise ValueError("static KV cache capacity does not match model block_size")

        x = self.input_projection(self.token_embedding(token_ids))
        if self.position_embedding is not None:
            positions = torch.arange(cache.length, end_position, device=token_ids.device)
            x = x + self.position_embedding(positions)[None, :, :]
        x = self.embedding_dropout(x)

        for index, block in enumerate(self.blocks):
            x = block.forward_with_static_cache(
                x,
                cache.keys[index],
                cache.values[index],
                cache_position=cache.length,
            )

        logits = self.output(self.output_projection(self.final_norm(x)))
        cache.length = end_position
        return logits, cache

    @torch.inference_mode()
    def generate(
        self,
        token_ids: torch.Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_token_id: int | None = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        if token_ids.ndim != 2 or token_ids.size(1) == 0:
            raise ValueError("token_ids must be a non-empty [batch, sequence] tensor")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero")
        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be positive when provided")
        if max_new_tokens == 0:
            return token_ids

        was_training = self.training
        self.eval()
        generated = token_ids
        try:
            if use_cache:
                context = generated[:, -self.config.block_size :]
                cache = self.allocate_static_kv_cache(
                    generated.size(0), device=generated.device
                )
                logits, cache = self.forward_with_static_cache(context, cache)
                next_logits = logits[:, -1, :] / temperature
                for step in range(max_new_tokens):
                    if top_k is not None:
                        k = min(top_k, next_logits.size(-1))
                        values, _ = torch.topk(next_logits, k)
                        next_logits = next_logits.masked_fill(
                            next_logits < values[:, [-1]], float("-inf")
                        )
                    probabilities = F.softmax(next_logits, dim=-1)
                    next_token = torch.multinomial(probabilities, num_samples=1)
                    generated = torch.cat((generated, next_token), dim=1)
                    if eos_token_id is not None and torch.all(next_token == eos_token_id):
                        break
                    if step + 1 < max_new_tokens:
                        if cache.length < self.config.block_size:
                            logits, cache = self.forward_with_static_cache(next_token, cache)
                        else:
                            # Reuse the same buffers for the latest sliding context.
                            context = generated[:, -self.config.block_size :]
                            cache.reset()
                            logits, cache = self.forward_with_static_cache(context, cache)
                        next_logits = logits[:, -1, :] / temperature
            else:
                for _ in range(max_new_tokens):
                    context = generated[:, -self.config.block_size :]
                    logits, _ = self(context)
                    next_logits = logits[:, -1, :] / temperature
                    if top_k is not None:
                        k = min(top_k, next_logits.size(-1))
                        values, _ = torch.topk(next_logits, k)
                        next_logits = next_logits.masked_fill(
                            next_logits < values[:, [-1]], float("-inf")
                        )
                    probabilities = F.softmax(next_logits, dim=-1)
                    next_token = torch.multinomial(probabilities, num_samples=1)
                    generated = torch.cat((generated, next_token), dim=1)
                    if eos_token_id is not None and torch.all(next_token == eos_token_id):
                        break
        finally:
            self.train(was_training)
        return generated

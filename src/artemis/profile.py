"""Profile Artemis parameter counts and estimated KV-cache memory."""
from __future__ import annotations

import argparse
import json

import torch

from artemis.config import ModelConfig
from artemis.model import ArtemisLM


def parameter_count(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def profile_model(
    config: ModelConfig,
    *,
    batch_size: int = 1,
    context_length: int | None = None,
    dtype_bytes: int = 2,
) -> dict[str, int | float | str | dict]:
    """Return unique parameter counts and an estimated compact KV-cache size.

    The KV-cache estimate assumes keys and values are stored using n_kv_head
    heads without expansion. Current Artemis generation does not yet implement
    an incremental cache, so this is a target-memory estimate, not measured use.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if dtype_bytes < 1:
        raise ValueError("dtype_bytes must be positive")
    context = config.block_size if context_length is None else context_length
    if context < 1 or context > config.block_size:
        raise ValueError("context_length must be between 1 and block_size")

    model = ArtemisLM(config)
    params_by_component: dict[str, int] = {
        "token_embedding_and_output_tied": parameter_count(model.token_embedding),
        "embedding_projections": (
            parameter_count(model.input_projection) + parameter_count(model.output_projection)
        ),
        "position_embedding": (
            parameter_count(model.position_embedding)
            if model.position_embedding is not None
            else 0
        ),
        "final_norm": parameter_count(model.final_norm),
    }
    attention_total = 0
    feed_forward_total = 0
    block_norms_total = 0
    for block in model.blocks:
        attention_total += parameter_count(block.attention)
        feed_forward_total += parameter_count(block.feed_forward)
        block_norms_total += parameter_count(block.norm1) + parameter_count(block.norm2)

    params_by_component["attention_all_layers"] = attention_total
    params_by_component["feed_forward_all_layers"] = feed_forward_total
    params_by_component["block_norms_all_layers"] = block_norms_total

    # Count unique named parameters to avoid double-counting tied embedding/head weights.
    total_parameters = sum(p.numel() for p in model.parameters())
    n_kv_head = config.n_head if config.n_kv_head is None else config.n_kv_head
    head_dim = config.n_embd // config.n_head
    estimated_kv_bytes = (
        batch_size
        * config.n_layer
        * context
        * 2
        * n_kv_head
        * head_dim
        * dtype_bytes
    )
    full_mha_kv_bytes = (
        batch_size
        * config.n_layer
        * context
        * 2
        * config.n_head
        * head_dim
        * dtype_bytes
    )
    return {
        "model_parameters": total_parameters,
        "parameters_by_component": params_by_component,
        "architecture": {
            "layers": config.n_layer,
            "query_heads": config.n_head,
            "kv_heads": n_kv_head,
            "head_dimension": head_dim,
            "embedding_width": config.n_embd,
            "vocabulary_size": config.vocab_size,
            "context_window": config.block_size,
            "use_rope": config.use_rope,
            "use_qk_norm": config.use_qk_norm,
            "use_swiglu": config.use_swiglu,
        },
        "kv_cache_estimate": {
            "batch_size": batch_size,
            "context_length": context,
            "dtype_bytes": dtype_bytes,
            "gqa_cache_bytes": estimated_kv_bytes,
            "full_mha_cache_bytes": full_mha_kv_bytes,
            "reduction_fraction_vs_full_mha": (
                0.0
                if full_mha_kv_bytes == 0
                else 1.0 - estimated_kv_bytes / full_mha_kv_bytes
            ),
            "note": (
                "Estimated tensor payload for the incremental cache implemented by "
                "forward_with_cache/generate. Actual peak memory also includes model "
                "weights, activations, temporary expanded GQA tensors, allocator overhead, "
                "and other runtime buffers."
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab-size", type=int, default=2048)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--kv-heads", type=int)
    parser.add_argument("--embedding-size", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, help="factorised token embedding width")
    parser.add_argument("--rope", action="store_true")
    parser.add_argument("--qk-norm", action="store_true")
    parser.add_argument("--swiglu", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--context-length", type=int)
    parser.add_argument("--dtype-bytes", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ModelConfig(
        vocab_size=args.vocab_size,
        block_size=args.block_size,
        n_layer=args.layers,
        n_head=args.heads,
        n_embd=args.embedding_size,
        dropout=args.dropout,
        embedding_dim=args.embedding_dim,
        n_kv_head=args.kv_heads,
        use_rope=args.rope,
        use_qk_norm=args.qk_norm,
        use_swiglu=args.swiglu,
    )
    print(json.dumps(
        profile_model(
            config,
            batch_size=args.batch_size,
            context_length=args.context_length,
            dtype_bytes=args.dtype_bytes,
        ),
        indent=2,
    ))


if __name__ == "__main__":
    main()

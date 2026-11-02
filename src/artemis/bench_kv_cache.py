"""Benchmark static, dynamic, and uncached Artemis decoding on random-init weights.

This is a systems benchmark, not a model-quality evaluation. All paths use the
same model instance, prompt, and deterministic top-k=1 decoding policy.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from artemis.config import ModelConfig
from artemis.model import ArtemisLM, StaticKVCache
from artemis.train import choose_device


def synchronise(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def cache_payload_bytes(cache: tuple[tuple[torch.Tensor, torch.Tensor], ...]) -> int:
    return sum(tensor.numel() * tensor.element_size() for layer in cache for tensor in layer)


def static_cache_payload_bytes(cache: StaticKVCache, *, active_only: bool) -> int:
    """Count active tensor payload or the full preallocated K/V capacity."""
    length = cache.length if active_only else cache.capacity
    return sum(
        tensor[:, :, :length, :].numel() * tensor.element_size()
        for tensor_group in (cache.keys, cache.values)
        for tensor in tensor_group
    )


def _timed_generation(
    model: ArtemisLM,
    prompt: torch.Tensor,
    *,
    new_tokens: int,
    cache_mode: str,
    device: torch.device,
) -> tuple[float, torch.Tensor]:
    if cache_mode not in {"static", "dynamic", "uncached"}:
        raise ValueError("cache_mode must be static, dynamic, or uncached")
    synchronise(device)
    start = time.perf_counter()
    output = model.generate(
        prompt,
        new_tokens,
        temperature=1.0,
        top_k=1,
        eos_token_id=None,
        use_cache=cache_mode != "uncached",
        use_static_cache=cache_mode == "static",
    )
    synchronise(device)
    return time.perf_counter() - start, output


def benchmark_generation(
    model: ArtemisLM,
    prompt: torch.Tensor,
    *,
    new_tokens: int,
    warmup: int = 1,
    repeats: int = 5,
    device: torch.device | None = None,
) -> dict[str, object]:
    if new_tokens < 0:
        raise ValueError("new_tokens must be non-negative")
    if warmup < 0 or repeats < 1:
        raise ValueError("warmup must be non-negative and repeats must be positive")
    if prompt.ndim != 2 or prompt.size(1) == 0:
        raise ValueError("prompt must be a non-empty [batch, sequence] tensor")

    selected_device = prompt.device if device is None else device
    if prompt.device != selected_device:
        raise ValueError("prompt tensor and benchmark device must match")
    if prompt.size(1) > model.config.block_size:
        prompt = prompt[:, -model.config.block_size :]

    model.eval()
    modes = ("static", "dynamic", "uncached")
    for _ in range(warmup):
        for mode in modes:
            _timed_generation(
                model, prompt, new_tokens=new_tokens,
                cache_mode=mode, device=selected_device,
            )

    times: dict[str, list[float]] = {mode: [] for mode in modes}
    outputs: dict[str, torch.Tensor] = {}
    for _ in range(repeats):
        for mode in modes:
            elapsed, output = _timed_generation(
                model, prompt, new_tokens=new_tokens,
                cache_mode=mode, device=selected_device,
            )
            times[mode].append(elapsed)
            outputs[mode] = output

    if any(mode not in outputs for mode in modes):
        raise RuntimeError("benchmark did not produce output for every cache mode")
    outputs_match = (
        torch.equal(outputs["static"], outputs["dynamic"])
        and torch.equal(outputs["dynamic"], outputs["uncached"])
    )

    static_cache = model.allocate_static_kv_cache(prompt.size(0), device=selected_device)
    with torch.inference_mode():
        _, static_cache = model.forward_with_static_cache(prompt, static_cache)
    active_cache_bytes = static_cache_payload_bytes(static_cache, active_only=True)
    allocated_cache_bytes = static_cache_payload_bytes(static_cache, active_only=False)

    batch_size = prompt.size(0)
    context_length = prompt.size(1)
    dtype_bytes = next(model.parameters()).element_size()
    full_mha_active_bytes = (
        batch_size
        * model.config.n_layer
        * context_length
        * 2
        * model.config.n_head
        * (model.config.n_embd // model.config.n_head)
        * dtype_bytes
    )
    medians = {mode: statistics.median(times[mode]) for mode in modes}

    def tokens_per_second(mode: str) -> float | None:
        elapsed = medians[mode]
        return new_tokens / elapsed if elapsed > 0 else None

    return {
        "weights": "random initialization; systems benchmark only",
        "device": str(selected_device),
        "batch_size": batch_size,
        "prompt_tokens": context_length,
        "generated_tokens": new_tokens,
        "warmup_runs": warmup,
        "timed_repeats": repeats,
        "static_cached_latency_ms": medians["static"] * 1000.0,
        "dynamic_cached_latency_ms": medians["dynamic"] * 1000.0,
        "uncached_latency_ms": medians["uncached"] * 1000.0,
        "static_cached_generated_tokens_per_second": tokens_per_second("static"),
        "dynamic_cached_generated_tokens_per_second": tokens_per_second("dynamic"),
        "uncached_generated_tokens_per_second": tokens_per_second("uncached"),
        "uncached_over_static_speed_ratio": (
            medians["uncached"] / medians["static"] if medians["static"] > 0 else None
        ),
        "uncached_over_dynamic_speed_ratio": (
            medians["uncached"] / medians["dynamic"] if medians["dynamic"] > 0 else None
        ),
        "dynamic_over_static_speed_ratio": (
            medians["static"] / medians["dynamic"] if medians["dynamic"] > 0 else None
        ),
        "static_dynamic_and_uncached_outputs_match": outputs_match,
        "kv_cache_active_payload_bytes": active_cache_bytes,
        "kv_cache_allocated_capacity_bytes": allocated_cache_bytes,
        "full_mha_active_cache_estimate_bytes": full_mha_active_bytes,
        "kv_cache_active_payload_reduction_fraction": (
            1.0 - active_cache_bytes / full_mha_active_bytes
            if full_mha_active_bytes > 0 else 0.0
        ),
        "warning": (
            "Latency includes prompt prefill and token sampling. Static cache reserves "
            "block_size capacity; dynamic cache grows to active length. Results depend "
            "on device, PyTorch kernels, context, and model configuration. Do not "
            "generalise from a random-init smoke model to trained-model quality."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--embedding-size", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--rope", action="store_true")
    parser.add_argument("--qk-norm", action="store_true")
    parser.add_argument("--swiglu", action="store_true")
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--new-tokens", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.prompt_tokens < 1 or args.prompt_tokens > args.block_size:
        raise SystemExit("prompt-tokens must be between 1 and block-size")
    if args.threads < 1:
        raise SystemExit("threads must be positive")

    device = choose_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    config = ModelConfig(
        vocab_size=args.vocab_size,
        block_size=args.block_size,
        n_layer=args.layers,
        n_head=args.heads,
        n_embd=args.embedding_size,
        n_kv_head=args.kv_heads,
        use_rope=args.rope,
        use_qk_norm=args.qk_norm,
        use_swiglu=args.swiglu,
        embedding_dim=args.embedding_dim,
    )
    model = ArtemisLM(config).to(device)
    prompt = torch.randint(0, config.vocab_size, (1, args.prompt_tokens), device=device)
    result = benchmark_generation(
        model,
        prompt,
        new_tokens=args.new_tokens,
        warmup=args.warmup,
        repeats=args.repeats,
        device=device,
    )
    result["model_parameters"] = sum(parameter.numel() for parameter in model.parameters())
    result["architecture"] = config.to_dict()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""Benchmark cached versus uncached Artemis decoding on random-init weights.

This is a systems benchmark, not a model-quality evaluation. Use the same model
instance, prompt, and decoding policy for both paths.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from artemis.config import ModelConfig
from artemis.model import ArtemisLM
from artemis.train import choose_device


def synchronise(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def cache_payload_bytes(cache: tuple[tuple[torch.Tensor, torch.Tensor], ...]) -> int:
    return sum(tensor.numel() * tensor.element_size() for layer in cache for tensor in layer)


def _timed_generation(
    model: ArtemisLM,
    prompt: torch.Tensor,
    *,
    new_tokens: int,
    use_cache: bool,
    device: torch.device,
) -> tuple[float, torch.Tensor]:
    synchronise(device)
    start = time.perf_counter()
    output = model.generate(
        prompt,
        new_tokens,
        temperature=1.0,
        top_k=1,
        eos_token_id=None,
        use_cache=use_cache,
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
    for _ in range(warmup):
        _timed_generation(
            model, prompt, new_tokens=new_tokens, use_cache=True, device=selected_device
        )
        _timed_generation(
            model, prompt, new_tokens=new_tokens, use_cache=False, device=selected_device
        )

    cached_times: list[float] = []
    uncached_times: list[float] = []
    cached_output = uncached_output = None
    for _ in range(repeats):
        elapsed, cached_output = _timed_generation(
            model, prompt, new_tokens=new_tokens, use_cache=True, device=selected_device
        )
        cached_times.append(elapsed)
        elapsed, uncached_output = _timed_generation(
            model, prompt, new_tokens=new_tokens, use_cache=False, device=selected_device
        )
        uncached_times.append(elapsed)

    if cached_output is None or uncached_output is None:
        raise RuntimeError("benchmark did not produce generation outputs")
    outputs_match = bool(torch.equal(cached_output, uncached_output))

    with torch.inference_mode():
        _, cache = model.forward_with_cache(prompt)
    cache_bytes = cache_payload_bytes(cache)
    batch_size = prompt.size(0)
    context_length = prompt.size(1)
    dtype_bytes = next(model.parameters()).element_size()
    estimated_full_mha_bytes = (
        batch_size
        * model.config.n_layer
        * context_length
        * 2
        * model.config.n_head
        * (model.config.n_embd // model.config.n_head)
        * dtype_bytes
    )

    cached_median = statistics.median(cached_times)
    uncached_median = statistics.median(uncached_times)
    return {
        "weights": "random initialization; systems benchmark only",
        "device": str(selected_device),
        "batch_size": batch_size,
        "prompt_tokens": context_length,
        "generated_tokens": new_tokens,
        "warmup_runs": warmup,
        "timed_repeats": repeats,
        "cached_latency_ms": cached_median * 1000.0,
        "uncached_latency_ms": uncached_median * 1000.0,
        "cached_generated_tokens_per_second": (
            new_tokens / cached_median if cached_median > 0 else None
        ),
        "uncached_generated_tokens_per_second": (
            new_tokens / uncached_median if uncached_median > 0 else None
        ),
        "uncached_over_cached_speed_ratio": (
            uncached_median / cached_median if cached_median > 0 else None
        ),
        "cached_and_uncached_outputs_match": outputs_match,
        "kv_cache_payload_bytes": cache_bytes,
        "full_mha_cache_estimate_bytes": estimated_full_mha_bytes,
        "kv_cache_payload_reduction_fraction": (
            1.0 - cache_bytes / estimated_full_mha_bytes
            if estimated_full_mha_bytes > 0
            else 0.0
        ),
        "warning": (
            "Latency includes prompt prefill and token sampling. Results depend on "
            "device, PyTorch kernels, context, and model configuration; do not generalise "
            "from a random-init smoke model to trained-model quality."
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
    prompt = torch.randint(
        0, config.vocab_size, (1, args.prompt_tokens), device=device
    )
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

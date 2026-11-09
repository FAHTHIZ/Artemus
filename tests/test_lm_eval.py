import json
import math

import torch

from artemis.config import ModelConfig
from artemis.lm_eval import evaluate_documents, read_documents
from artemis.model import ArtemisLM
from artemis.tokenizer import ByteTokenizer


def test_read_documents_respects_limit_and_skips_blank(tmp_path):
    path = tmp_path / "v.jsonl"
    path.write_text("\n".join(json.dumps({"text": t}) for t in ["a", "", "b", "c"]) + "\n")
    assert read_documents(path, limit=2) == ["a", "b"]


def test_uniform_model_bits_per_byte_matches_theory():
    torch.manual_seed(0)
    config = ModelConfig(vocab_size=257, block_size=16, n_layer=1, n_head=2, n_embd=16)
    model = ArtemisLM(config)
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    text = "hello world, this is a held-out document."
    result = evaluate_documents(model, ByteTokenizer(), [text], block_size=16)
    # All-zero weights => uniform logits => loss = ln(257) per token.
    assert math.isclose(result["token_loss"], math.log(257), rel_tol=1e-4)
    tokens = len(text.encode()) + 1 - 1  # bytes + EOS, minus the first unpredicted token
    expected_bpb = tokens * math.log2(257) / len(text.encode())
    assert math.isclose(result["bits_per_byte"], expected_bpb, rel_tol=1e-4)
    assert result["documents"] == 1

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from train_lora import answer_loss  # noqa: E402


def test_answer_loss_matches_full_loss():
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(
        vocab_size=500,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
    )
    model = transformers.LlamaForCausalLM(cfg).eval()
    ids = torch.randint(0, 500, (1, 40))
    labels = ids.clone()
    labels[:, :27] = -100
    mask = torch.ones_like(ids)
    with torch.no_grad():
        ref = model(input_ids=ids, attention_mask=mask, labels=labels).loss
        new = answer_loss(model, {"input_ids": ids, "attention_mask": mask, "labels": labels})
    assert torch.allclose(ref, new, atol=1e-5)

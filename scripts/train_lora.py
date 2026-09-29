"""M5: LoRA fine-tuning of a Qwen-VL-class model on the crash-report instruction data (GPU).

    python scripts/train_lora.py vlm.model=Qwen/Qwen3.5-4B vlm.run_name=qwen3.5-4b_lora
    python scripts/train_lora.py sft.train_file=train_f0.25.jsonl vlm.run_name=qwen3.5-4b_lora_f0.25

Plain PyTorch + PEFT loop (batch size 1, gradient accumulation), loss on the answer tokens only
(the prompt, including all image tokens, is masked with -100). Validation loss after every
epoch; the adapter with the lowest validation loss is kept. Writes the adapter and
train_log.json to ``sft.output_dir``.
"""

import io
import json
import math
import random
import time
import zipfile
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from crashvlm.models.vlm_report import build_messages


def encode(processor, ex, zf, max_side):
    ims = []
    for f in ex["files"]:
        im = Image.open(io.BytesIO(zf.read(f))).convert("RGB")
        s = max_side / max(im.size)
        ims.append(im.resize((round(im.width * s), round(im.height * s))) if s < 1 else im)
    msgs = build_messages(ims, ex["rel_ms"], ex["role"])
    msgs.append({"role": "assistant", "content": [{"type": "text", "text": ex["target"]}]})
    try:
        text = processor.apply_chat_template(msgs, tokenize=False, enable_thinking=False)
    except TypeError:
        text = processor.apply_chat_template(msgs, tokenize=False)
    # Mask everything before the answer. Cutting the rendered text works for any template
    # (with or without an empty <think></think> block before the answer).
    cut = text.rindex(ex["target"])
    full = processor(text=[text], images=ims, return_tensors="pt")
    prefix = processor(text=[text[:cut]], images=ims, return_tensors="pt")
    n_prompt = prefix["input_ids"].shape[1]
    if not bool((full["input_ids"][0, :n_prompt] == prefix["input_ids"][0]).all()):
        raise ValueError("prompt tokens are not a prefix of prompt+answer tokens")
    labels = full["input_ids"].clone()
    labels[:, :n_prompt] = -100
    full["labels"] = labels
    return full


def answer_loss(model, batch):
    """Cross-entropy on the answer tokens only, without materialising logits for the prompt.

    The full logits (about 3000 positions x a 248k-token vocabulary) do not fit on a 16 GB GPU
    next to the model. ``logits_to_keep`` makes the model compute only the last k positions:
    the one before the first answer token and the answer itself."""
    import torch.nn.functional as F

    batch = dict(batch)
    labels = batch.pop("labels")
    n_prompt = int((labels[0] == -100).sum())
    k = labels.shape[1] - n_prompt + 1
    logits = model(**batch, logits_to_keep=k).logits[:, -k:-1]
    target = batch["input_ids"][:, n_prompt:]
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), target.reshape(-1))


@hydra.main(config_path="../configs", config_name="m5", version_base="1.3")
def main(cfg: DictConfig) -> None:
    import os

    from crashvlm.models.compat import hide_old_torchao

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    hide_old_torchao()
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor

    s, v = cfg.sft, cfg.vlm
    random.seed(cfg.m3.seed)
    torch.manual_seed(cfg.m3.seed)
    sft_dir = Path(cfg.paths.root) / "sft"
    train = [json.loads(line) for line in (sft_dir / s.train_file).read_text().splitlines()]
    val = [json.loads(line) for line in (sft_dir / "val.jsonl").read_text().splitlines()]
    if s.val_limit:
        val = val[: s.val_limit]
    out = Path(s.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cuda = torch.cuda.is_available()
    bf16 = cuda and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else (torch.float16 if cuda else torch.float32)
    kwargs = {"torch_dtype": dtype}
    if v.load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype
        )
    if cuda:
        kwargs["device_map"] = "auto"
    model = AutoModelForImageTextToText.from_pretrained(v.model, **kwargs)
    processor = AutoProcessor.from_pretrained(v.model)
    if v.load_in_4bit:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model, s.gradient_checkpointing)
    elif s.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    lcfg = LoraConfig(
        r=s.lora_r,
        lora_alpha=s.lora_alpha,
        lora_dropout=s.lora_dropout,
        target_modules=list(s.target_modules),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lcfg)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {n_train / 1e6:.1f} M of {n_all / 1e6:.0f} M")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=s.lr, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(train) / s.grad_accum)
    total = s.max_steps if s.max_steps > 0 else steps_per_epoch * s.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(1.0, (i + 1) / max(1, total // 20)) * max(0.0, 1 - i / total)
    )
    device = next(model.parameters()).device
    log = {"config": OmegaConf.to_container(cfg, resolve=True), "steps": [], "val": []}
    best = math.inf
    step = 0
    t_start = time.time()
    zf = zipfile.ZipFile(cfg.annotation.images_zip)

    def write_log() -> None:  # after every epoch, so a killed run still leaves a log
        log["best_val_loss"] = best
        log["train_minutes"] = (time.time() - t_start) / 60
        if cuda:
            log["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 2**30
        (out / "train_log.json").write_text(json.dumps(log, indent=1))

    def val_loss() -> float:
        model.eval()
        losses = []
        with torch.no_grad():
            for ex in val:
                batch = encode(processor, ex, zf, s.max_side).to(device)
                losses.append(float(answer_loss(model, batch)))
        model.train()
        return sum(losses) / len(losses)

    model.train()
    done = False
    for epoch in range(s.epochs):
        random.shuffle(train)
        running = 0.0
        for i, ex in enumerate(train):
            batch = encode(processor, ex, zf, s.max_side).to(device)
            loss = answer_loss(model, batch) / s.grad_accum
            loss.backward()
            running += float(loss.detach())
            if (i + 1) % s.grad_accum == 0 or i + 1 == len(train):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                log["steps"].append({"step": step, "loss": running, "lr": sched.get_last_lr()[0]})
                if step % 10 == 0 or step == 1:
                    print(f"epoch {epoch} step {step}/{total} loss {running:.4f}", flush=True)
                running = 0.0
                if s.max_steps > 0 and step >= s.max_steps:
                    done = True
                    break
                if s.time_budget_h and time.time() - t_start > s.time_budget_h * 3600:
                    print(f"time budget of {s.time_budget_h} h reached at step {step}")
                    log["stopped_by_time_budget"] = True
                    done = True
                    break
        vl = val_loss()
        log["val"].append({"epoch": epoch, "step": step, "val_loss": vl})
        print(f"epoch {epoch}: val loss {vl:.4f}", flush=True)
        if vl < best:
            best = vl
            model.save_pretrained(out / "adapter")
        write_log()
        if done:
            break
    write_log()
    print(f"best val loss {best:.4f}; adapter -> {out / 'adapter'}")


if __name__ == "__main__":
    main()

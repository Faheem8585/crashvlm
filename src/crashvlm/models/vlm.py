"""Hugging Face image-text-to-text runner for Qwen-VL-class models (GPU; ``ml`` extra).

One code path for Qwen3.5 (native multimodal) and Qwen3-VL: chat template with interleaved
images, greedy decoding, thinking disabled where the template supports it. Records latency and
peak GPU memory per call, so M6 can report efficiency from the same runs.
"""

from __future__ import annotations

import time


class VLMRunner:
    def __init__(
        self,
        model_id: str,
        load_in_4bit: bool = False,
        max_side: int = 384,
        max_new_tokens: int = 512,
        adapter: str | None = None,
    ):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        kwargs: dict = {"torch_dtype": torch.bfloat16 if self.device == "cuda" else torch.float32}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        if self.device == "cuda":
            kwargs["device_map"] = "auto"
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs).eval()
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.max_side = max_side
        self.max_new_tokens = max_new_tokens

    def _resize(self, im):
        s = self.max_side / max(im.size)
        if s < 1:
            im = im.resize((round(im.width * s), round(im.height * s)))
        return im

    def generate(self, messages: list[dict]) -> dict:
        for m in messages:
            for c in m["content"]:
                if c["type"] == "image":
                    c["image"] = self._resize(c["image"])
        try:
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                enable_thinking=False,
            )
        except TypeError:  # templates without the thinking switch
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        inputs = inputs.to(self.model.device)
        if self.device == "cuda":
            self.torch.cuda.reset_peak_memory_stats()
            self.torch.cuda.synchronize()
        t0 = time.perf_counter()
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        if self.device == "cuda":
            self.torch.cuda.synchronize()
        latency = time.perf_counter() - t0
        n_in = inputs["input_ids"].shape[1]
        text = self.processor.batch_decode(out[:, n_in:], skip_special_tokens=True)[0]
        peak = self.torch.cuda.max_memory_allocated() / 2**30 if self.device == "cuda" else None
        return {
            "text": text,
            "latency_s": latency,
            "peak_vram_gb": peak,
            "n_input_tokens": int(n_in),
            "n_output_tokens": int(out.shape[1] - n_in),
        }

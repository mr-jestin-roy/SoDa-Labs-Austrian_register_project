"""VLM client with two interchangeable backends.

1. OpenAI-compatible HTTP (default) — works unchanged against:
     * the existing Nectar vLLM Chandra endpoint  (VLM_MODEL=chandra)
     * a vLLM-served Qwen3-VL                      (VLM_MODEL=Qwen/Qwen3-VL-8B-Instruct)
     * any hosted multimodal API                   (set VLM_BASE_URL / VLM_API_KEY)

2. Local MLX (Apple Silicon) — selected automatically when the model id
   starts with "mlx-community/" (or VLM_BACKEND=mlx), e.g.
     VLM_MODEL=mlx-community/Qwen3-VL-4B-Instruct-4bit
   Runs Qwen3-VL on-device via mlx-vlm; no server needed.

Images are downscaled to MAX_SEND_DIM on the longest edge before encoding —
this is the vision-token-overflow mitigation from May13thHandoff.md.
"""
import base64
import io
import os
import tempfile
from pathlib import Path

from openai import OpenAI
from PIL import Image

import config


def image_to_data_url(image_path: str | Path, max_dim: int = config.MAX_SEND_DIM) -> str:
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    scale = min(max_dim / max(w, h), 1.0)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.standard_b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


class MLXVLMClient:
    """On-device Qwen3-VL via mlx-vlm (Apple Silicon). Loads lazily so that
    importing this module never pulls weights."""

    def __init__(self, model: str | None = None):
        self.model_id = model or config.VLM_MODEL
        self._loaded = None

    def _load(self):
        if self._loaded is None:
            from mlx_vlm import load
            from mlx_vlm.utils import load_config
            model, processor = load(self.model_id)
            self._loaded = (model, processor, load_config(self.model_id))
        return self._loaded

    def transcribe(self, image_path: str | Path, prompt: str,
                   max_tokens: int = config.VLM_MAX_TOKENS) -> str:
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template
        model, processor, cfg = self._load()
        # pre-shrink the page so vision tokens fit in 8 GB unified memory
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        scale = min(config.MAX_SEND_DIM / max(w, h), 1.0)
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            img.save(tmp.name)
            formatted = apply_chat_template(processor, cfg, prompt,
                                            num_images=1)
            out = generate(model, processor, formatted, image=[tmp.name],
                           max_tokens=max_tokens, temperature=0.0,
                           verbose=False)
        os.unlink(tmp.name)
        return getattr(out, "text", out)


class VLMClient:
    def __new__(cls, model: str | None = None, base_url: str | None = None):
        chosen = model or config.VLM_MODEL
        if (os.getenv("VLM_BACKEND") == "mlx"
                or chosen.startswith("mlx-community/")):
            return MLXVLMClient(chosen)
        return super().__new__(cls)

    def __init__(self, model: str | None = None, base_url: str | None = None):
        self.model = model or config.VLM_MODEL
        self.client = OpenAI(base_url=base_url or config.VLM_BASE_URL,
                             api_key=config.VLM_API_KEY)

    def transcribe(self, image_path: str | Path, prompt: str,
                   max_tokens: int = config.VLM_MAX_TOKENS) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=config.VLM_TEMPERATURE,
            max_tokens=max_tokens,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": image_to_data_url(image_path)}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        return resp.choices[0].message.content or ""

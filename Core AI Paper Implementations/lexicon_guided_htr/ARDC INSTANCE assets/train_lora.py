"""LoRA fine-tune of datalab-to/chandra-ocr-2 (Qwen3_5ForConditionalGeneration)
on the Reichstelefonbuch manifest produced by data_prep.py.

Trains the model to reliably emit the frozen page-level JSON schema (see
data_prep.py's PAGE_PASS1_PROMPT) instead of its native bbox-HTML format —
prompting alone could not get it to do this (see rtb_benchmark.py's
parse_chandra_html + the RTB p.6 benchmark: 0 masks emitted, schema
ignored). Loss is masked to assistant tokens only.

Usage:
    venv/bin/python train_lora.py --manifest /mnt/finetune/data/manifest.jsonl \
        --output-dir /mnt/finetune/checkpoints/run1 \
        --epochs 2 --batch-size 1 --grad-accum 8
"""
import argparse
import json
import random
from pathlib import Path

import torch
from PIL import Image
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForImageTextToText, AutoProcessor,
                          Trainer, TrainingArguments)

MODEL_ID = "datalab-to/chandra-ocr-2"

# Attention + MLP projections of the Qwen3.5 hybrid (linear-attention /
# full-attention) text backbone. Vision tower is left frozen — this is a
# text-generation-format fix (JSON compliance + field accuracy), not a
# glyph-recognition fix, so there is no reason to touch the ViT/merger.
# Names verified against transformers' modeling_qwen3_5.py (not guessed):
# Qwen3_5Attention (8 of 32 layers, full_attention_interval=4) uses
# q/k/v/o_proj; Qwen3_5GatedDeltaNet (the other 24, linear-attention/
# mamba-style) uses a DIFFERENT split — in_proj_qkv/in_proj_z/in_proj_b/
# in_proj_a plus out_proj (not "in_proj"/"o_proj").
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",                     # full attention
    "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",  # GatedDeltaNet
    "gate_proj", "up_proj", "down_proj",                          # MLP, all layers
]


def load_manifest(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


class RTBDataset(torch.utils.data.Dataset):
    def __init__(self, examples: list[dict]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def make_collate_fn(processor):
    def collate(batch: list[dict]):
        # single-example batches (batch_size=1, grad_accum for effective
        # batch size) — keeps variable-length page images/targets simple
        ex = batch[0]
        image = Image.open(ex["image"]).convert("RGB")
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": ex["prompt"]},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": ex["target"]},
            ]},
        ]
        # tokenize the user turn alone first, to find where the assistant
        # turn starts (for loss masking)
        prompt_only = processor.apply_chat_template(
            messages[:1], add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt")
        full = processor.apply_chat_template(
            messages, add_generation_prompt=False, tokenize=True,
            return_dict=True, return_tensors="pt")

        input_ids = full["input_ids"]
        labels = input_ids.clone()
        prompt_len = prompt_only["input_ids"].shape[1]
        labels[:, :prompt_len] = -100  # mask everything except the target JSON

        out = {k: v for k, v in full.items()}
        out["labels"] = labels
        return out

    def collate_batch(batch):
        # Trainer expects a single dict of batched tensors; since we force
        # batch_size=1 at the DataLoader level, just unwrap.
        return collate(batch)

    return collate_batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--val-fraction", type=float, default=0.05)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--load-in-4bit", action="store_true",
                    help="QLoRA-style 4-bit base weights; saves VRAM at "
                         "some quality cost. Off by default — full-precision "
                         "bf16 LoRA is the intended path.")
    ap.add_argument("--max-target-chars", type=int, default=28000,
                    help="Drop training examples whose target JSON exceeds "
                         "this many characters. With Chandra's vocab of "
                         "248,320 the cross-entropy loss allocates a fp32 "
                         "tensor of shape [seq_len, vocab] — a full page "
                         "with 200+ entries can push seq_len past 15k and "
                         "OOM a 40GB A100. 28k chars ≈ 10k tokens ≈ safe.")
    args = ap.parse_args()

    examples = load_manifest(args.manifest)
    n_before = len(examples)
    examples = [e for e in examples if len(e["target"]) <= args.max_target_chars]
    dropped = n_before - len(examples)
    if dropped:
        print(f"filter: dropped {dropped}/{n_before} examples with target > "
              f"{args.max_target_chars} chars (dense front-matter pages)")
    rng = random.Random(args.seed)
    rng.shuffle(examples)
    n_val = max(1, int(len(examples) * args.val_fraction))
    val_examples, train_examples = examples[:n_val], examples[n_val:]
    print(f"train={len(train_examples)} val={len(val_examples)}")

    processor = AutoProcessor.from_pretrained(MODEL_ID)

    model_kwargs = dict(dtype=torch.bfloat16, device_map="auto")
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4")
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, **model_kwargs)
    model.config.use_cache = False
    if args.load_in_4bit:
        # Correctly plumb a bitsandbytes-quantized model for LoRA training:
        # keeps LayerNorms in fp32, marks quantized params non-trainable, and
        # (crucially here) prevents accelerate's later `.to(device)` call
        # from erroring on quantized weights.
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True)

    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        target_modules=LORA_TARGET_MODULES, task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        report_to=["tensorboard"],
        remove_unused_columns=False,
        gradient_checkpointing=True,
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=RTBDataset(train_examples),
        eval_dataset=RTBDataset(val_examples),
        data_collator=make_collate_fn(processor),
    )
    trainer.train()
    trainer.save_model(str(args.output_dir / "final"))
    processor.save_pretrained(str(args.output_dir / "final"))
    print(f"Saved LoRA adapter to {args.output_dir / 'final'}")


if __name__ == "__main__":
    main()

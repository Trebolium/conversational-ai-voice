"""QLoRA fine-tune Qwen2.5-3B-Instruct on a CUDA GPU (Colab T4, AWS, etc).

Counterpart to ../mlx/lora_config.yaml for machines with an NVIDIA GPU,
where mlx-lm (Apple Silicon only) can't run. Uses the standard
transformers + peft + bitsandbytes + trl QLoRA stack. Install deps first:

    pip install -r requirements-cloud.txt

Then generate dataset/finetune/data/{train,valid}.jsonl (via
../prepare_data.py) before running this script.

Defaults are tuned conservatively for a single 16GB T4 -- bnb_4bit_compute
dtype is fp16 rather than bf16 since T4 (compute capability 7.5) lacks
bf16 tensor-core support (Ampere/8.0+ only). If running on a newer GPU
(A10/A100/L4), pass --bf16 instead.
"""

from __future__ import annotations

import argparse

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_TRAIN_FILE = "dataset/finetune/data/train.jsonl"
DEFAULT_VALID_FILE = "dataset/finetune/data/valid.jsonl"
DEFAULT_OUTPUT_DIR = "dataset/finetune/cloud/adapters"

TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA fine-tune Qwen2.5-3B-Instruct.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--train-file", default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--valid-file", default=DEFAULT_VALID_FILE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bf16", action="store_true", help="Use bf16 (Ampere+ GPUs) instead of fp16 (T4).")
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--num-train-epochs", type=float, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU detected -- this script requires a CUDA GPU (e.g. Colab T4).")

    compute_dtype = torch.bfloat16 if args.bf16 else torch.float16

    train_ds = load_dataset("json", data_files=args.train_file, split="train")
    eval_ds = load_dataset("json", data_files=args.valid_file, split="train")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        dtype=compute_dtype,
        device_map="auto",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )
    model = get_peft_model(model, lora_config)

    training_args = SFTConfig(
        output_dir=args.output_dir,
        max_length=args.max_length,
        packing=False,
        assistant_only_loss=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=0.03,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=not args.bf16,
        bf16=args.bf16,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()

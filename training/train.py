"""Phase 5 — QLoRA fine-tuning of Mistral 7B Instruct v0.2.

Reads config from training/config.yaml.
Trains on data/processed/train.json, evaluates on val.json.
Saves LoRA adapter checkpoints to ./checkpoints/.
Optionally pushes merged model to HuggingFace Hub.

Run on Colab (A100 or T4):
    python training/train.py
    python training/train.py --dry-run        # 20 examples, 1 epoch
    python training/train.py --push-to-hub    # push adapter after training
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def normalize_training_config(train_cfg: dict) -> dict:
    """Normalize training config types after YAML load.

    Some YAML scientific-notation values (e.g. 2e-4) may be loaded as strings
    depending on parser/version. Coerce known numeric fields explicitly.
    """
    int_keys = [
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "logging_steps",
        "save_total_limit",
        "dataloader_num_workers",
        "seed",
    ]
    float_keys = [
        "num_train_epochs",
        "learning_rate",
        "warmup_ratio",
        "weight_decay",
    ]

    for key in int_keys:
        if key in train_cfg:
            train_cfg[key] = int(train_cfg[key])

    for key in float_keys:
        if key in train_cfg:
            train_cfg[key] = float(train_cfg[key])

    return train_cfg


def resolve_precision(train_cfg: dict) -> tuple[bool, bool, torch.dtype]:
    """Resolve fp16/bf16 flags for the current GPU safely.

    On Ampere+ GPUs (A100/L4), bf16 is preferred because it avoids fp16
    grad-scaler edge cases in some accelerator/torch combinations.
    """
    use_fp16 = bool(train_cfg.get("fp16", False))
    use_bf16 = bool(train_cfg.get("bf16", False))

    if use_fp16 and use_bf16:
        raise ValueError("Config error: both fp16 and bf16 are enabled. Choose only one.")

    torch_dtype = torch.float32

    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability(0)
        ampere_or_newer = major >= 8

        if ampere_or_newer:
            # Prefer bf16 on Ampere+ when explicitly requested, otherwise keep fp16.
            if train_cfg.get("bf16", False):
                use_bf16 = True
                use_fp16 = False
        else:
            # T4-class GPUs are the problem case: disable mixed precision scaler logic
            # entirely to avoid bf16/fp16 autocast mismatches in the Colab stack.
            print("T4-class GPU detected. Disabling fp16/bf16 mixed precision for stability.")
            use_fp16 = False
            use_bf16 = False

    if use_bf16:
        torch_dtype = torch.bfloat16
    elif use_fp16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float16

    return use_fp16, use_bf16, torch_dtype


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_split(path: Path, limit: int | None = None, text_field: str = "text") -> Dataset:
    with path.open() as f:
        records = json.load(f)
    if limit:
        records = records[:limit]
    dataset = Dataset.from_list(records)
    if text_field not in dataset.column_names:
        raise KeyError(f"Missing required training field: {text_field}")

    # Keep only the flat text field so SFTTrainer uses language-modeling mode.
    return dataset.select_columns([text_field])


# ---------------------------------------------------------------------------
# Model + tokenizer
# ---------------------------------------------------------------------------

def load_tokenizer(model_name: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"   # required for SFTTrainer
    return tokenizer


def load_model(model_name: str, quant_cfg: dict, model_dtype: torch.dtype) -> AutoModelForCausalLM:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_cfg["load_in_4bit"],
        bnb_4bit_quant_type=quant_cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, quant_cfg["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=quant_cfg["bnb_4bit_use_double_quant"],
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=model_dtype,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    return model


def apply_lora(model, lora_cfg: dict, model_dtype: torch.dtype):
    config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
    )
    model = get_peft_model(model, config)

    # Keep all trainable adapter weights on the resolved precision to avoid
    # mixed bf16/fp16 gradient scaling issues on T4.
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.to(model_dtype)

    model.print_trainable_parameters()
    return model


# ---------------------------------------------------------------------------
# Logging callback
# ---------------------------------------------------------------------------

class LossLoggerCallback(TrainerCallback):
    """Logs train and eval loss to a CSV for easy plotting."""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.rows: list[dict] = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            row = {"step": state.global_step}
            if "loss" in logs:
                row["train_loss"] = round(logs["loss"], 4)
            if "eval_loss" in logs:
                row["eval_loss"] = round(logs["eval_loss"], 4)
            if len(row) > 1:
                self.rows.append(row)

    def on_train_end(self, args, state, control, **kwargs):
        import csv
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["step", "train_loss", "eval_loss"]
            )
            writer.writeheader()
            for row in self.rows:
                writer.writerow(row)
        print(f"Loss log saved → {self.log_path}")


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Train on 20 examples for 1 epoch to verify pipeline end-to-end.",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push LoRA adapter to HuggingFace Hub after training.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "training" / "config.yaml"),
        help="Path to config.yaml",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))

    model_name = cfg["model"]["name"]
    max_seq_length = cfg["model"]["max_seq_length"]
    train_cfg = normalize_training_config(cfg["training"])
    dataset_cfg = cfg["dataset"]
    use_fp16, use_bf16, model_dtype = resolve_precision(train_cfg)

    # dry-run overrides
    if args.dry_run:
        print("\nDRY RUN — 20 examples, 1 epoch\n")
        train_cfg["num_train_epochs"] = 1
        train_cfg["logging_steps"] = 1
        train_cfg["save_strategy"] = "no"
        train_cfg["eval_strategy"] = "no"
        train_cfg["load_best_model_at_end"] = False

    limit = 20 if args.dry_run else None

    # ---- data ------------------------------------------------------------
    print("Loading datasets...")
    train_dataset = load_split(
        ROOT / dataset_cfg["train_path"],
        limit=limit,
        text_field=dataset_cfg["text_field"],
    )
    val_dataset = load_split(
        ROOT / dataset_cfg["val_path"],
        limit=limit,
        text_field=dataset_cfg["text_field"],
    )
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    # ---- model -----------------------------------------------------------
    print(f"\nLoading {model_name}...")
    tokenizer = load_tokenizer(model_name)
    model = load_model(model_name, cfg["quantization"], model_dtype)
    model = apply_lora(model, cfg["lora"], model_dtype)
    print(f"Model dtype resolved to: {model_dtype}")

    vram_used = torch.cuda.memory_allocated() / 1e9
    print(f"VRAM after model load: {vram_used:.2f} GB")

    # ---- trainer ---------------------------------------------------------
    output_dir = ROOT / train_cfg["output_dir"]
    log_path = output_dir / "loss_log.csv"

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"],
        logging_steps=train_cfg["logging_steps"],
        save_strategy=train_cfg["save_strategy"],
        save_total_limit=train_cfg["save_total_limit"],
        eval_strategy=train_cfg["eval_strategy"],
        load_best_model_at_end=train_cfg["load_best_model_at_end"],
        metric_for_best_model=train_cfg["metric_for_best_model"],
        greater_is_better=train_cfg["greater_is_better"],
        fp16=use_fp16,
        bf16=use_bf16,
        optim="paged_adamw_8bit",
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        dataloader_num_workers=train_cfg["dataloader_num_workers"],
        report_to=train_cfg["report_to"],
        seed=train_cfg["seed"],
        dataset_text_field=dataset_cfg["text_field"],
        max_length=max_seq_length,
    )

    # Force flat language-modeling schema and guard against stale Colab files.
    text_field = dataset_cfg["text_field"]
    train_dataset = train_dataset.select_columns([text_field])
    val_dataset = val_dataset.select_columns([text_field])

    print("Train columns:", train_dataset.column_names)
    print("Val columns:", val_dataset.column_names)
    if train_dataset.column_names != [text_field] or val_dataset.column_names != [text_field]:
        raise RuntimeError(
            f"Unexpected dataset schema for SFTTrainer. Expected only ['{text_field}'], "
            f"got train={train_dataset.column_names}, val={val_dataset.column_names}"
        )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=sft_config,
        processing_class=tokenizer,
        callbacks=[LossLoggerCallback(log_path)],
    )

    # ---- train -----------------------------------------------------------
    print("\nStarting training...")
    trainer.train()

    # ---- save best checkpoint --------------------------------------------
    best_dir = output_dir / "best"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    print(f"\nBest adapter saved → {best_dir}")

    # ---- push to hub (optional) ------------------------------------------
    if args.push_to_hub:
        hub_id = cfg["hub"]["hub_model_id"]
        print(f"\nPushing adapter to HuggingFace Hub: {hub_id}")
        model.push_to_hub(hub_id)
        tokenizer.push_to_hub(hub_id)
        print("Push complete.")

    # ---- cleanup ---------------------------------------------------------
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"\nVRAM after cleanup: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print("\nPhase 5 training complete.")


if __name__ == "__main__":
    main()
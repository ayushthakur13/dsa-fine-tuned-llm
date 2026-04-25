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


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_split(path: Path, limit: int | None = None) -> Dataset:
    with path.open() as f:
        records = json.load(f)
    if limit:
        records = records[:limit]
    return Dataset.from_list(records)


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


def load_model(model_name: str, quant_cfg: dict) -> AutoModelForCausalLM:
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
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    return model


def apply_lora(model, lora_cfg: dict):
    config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
    )
    model = get_peft_model(model, config)
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
    train_cfg = cfg["training"]
    dataset_cfg = cfg["dataset"]

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
    train_dataset = load_split(ROOT / dataset_cfg["train_path"], limit=limit)
    val_dataset = load_split(ROOT / dataset_cfg["val_path"], limit=limit)
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    # ---- model -----------------------------------------------------------
    print(f"\nLoading {model_name}...")
    tokenizer = load_tokenizer(model_name)
    model = load_model(model_name, cfg["quantization"])
    model = apply_lora(model, cfg["lora"])

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
        fp16=train_cfg["fp16"],
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        dataloader_num_workers=train_cfg["dataloader_num_workers"],
        report_to=train_cfg["report_to"],
        seed=train_cfg["seed"],
        dataset_text_field=dataset_cfg["text_field"],
        max_length=max_seq_length,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=sft_config,
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
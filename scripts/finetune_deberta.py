"""Fine-tune DeBERTa-v3-large on WildGuardMix for binary safety classification.

Target: Mac Studio M3 Ultra (MPS) at 169.254.1.2
Dataset: allenai/wildguardmix from HuggingFace
Task: binary sequence classification (safe=0, unsafe=1)
Output: checkpoints/deberta-wildguardmix/
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


def get_device() -> str:
    """Select MPS if available, else CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def compute_metrics(eval_pred):
    """Compute F1, precision, recall, AUC for binary classification."""
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    preds = np.argmax(logits, axis=-1)
    return {
        "f1": f1_score(labels, preds, average="binary"),
        "precision": precision_score(labels, preds, average="binary"),
        "recall": recall_score(labels, preds, average="binary"),
        "auc": roc_auc_score(labels, probs[:, 1]),
    }


def main(device_override: str | None = None, max_steps: int = -1):
    model_name = "microsoft/deberta-v3-large"
    output_dir = "checkpoints/deberta-wildguardmix"
    device = device_override or get_device()
    print(f"Device: {device}")
    if max_steps > 0:
        print(f"Max steps: {max_steps} (validation run)")

    # Load dataset
    print("Loading WildGuardMix...")
    ds = load_dataset("allenai/wildguardmix", "wildguardtrain")
    train_ds = ds["train"]

    # Map labels: prompt_harm_label 'harmful' → 1, 'unharmful' → 0
    # Text field: 'prompt'
    def map_labels(example):
        raw = example["prompt_harm_label"]
        example["label"] = int(1 if raw == "harmful" else 0)
        example["text"] = example["prompt"]
        return example

    train_ds = train_ds.map(map_labels)

    # Shuffle before split for representative sampling
    train_ds = train_ds.shuffle(seed=42)

    # Train/eval split
    split = train_ds.train_test_split(test_size=0.1, seed=42)
    train_split = split["train"]
    eval_split = split["test"]
    print(f"Train: {len(train_split)}, Eval: {len(eval_split)}")

    # Verify labels are correct integers
    sample_labels = train_split[:8]["label"]
    print(f"Sample labels (first 8): {sample_labels}")
    assert all(isinstance(l, int) and l in (0, 1) for l in sample_labels), (
        f"Labels must be integers 0 or 1, got: {sample_labels}"
    )

    # Class balance check
    first_100 = train_split[:100]["label"]
    n_safe = first_100.count(0)
    n_unsafe = first_100.count(1)
    print(f"Class balance (first 100): safe={n_safe}, unsafe={n_unsafe}")
    if n_safe == 0 or n_unsafe == 0:
        raise RuntimeError(
            f"Training set has only one class in first 100 examples (safe={n_safe}, unsafe={n_unsafe}). "
            "Label mapping from WildGuardMix is likely wrong — check the 'safety_label' field values."
        )
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(examples):
        return tokenizer(
            examples["text"], truncation=True, max_length=512, padding="max_length"
        )

    train_split = train_split.map(tokenize, batched=True, remove_columns=[
        c for c in train_split.column_names if c not in ("label", "input_ids", "attention_mask")
    ])
    eval_split = eval_split.map(tokenize, batched=True, remove_columns=[
        c for c in eval_split.column_names if c not in ("label", "input_ids", "attention_mask")
    ])

    train_split.set_format("torch")
    eval_split.set_format("torch")

    # Model — initialize on CPU in fp32 (checkpoint may be fp16 which causes NaN)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, torch_dtype=torch.float32
    )

    # Sanity check: one forward pass on CPU before training
    print("Running sanity check forward pass (CPU)...")
    sample = train_split[0]
    model.eval()
    with torch.no_grad():
        input_ids = sample["input_ids"].unsqueeze(0)
        attention_mask = sample["attention_mask"].unsqueeze(0)
        labels = sample["label"].unsqueeze(0).long()

        # Debug: inspect inputs
        vocab_size = model.config.vocab_size
        print(f"  input_ids.shape: {input_ids.shape}, dtype: {input_ids.dtype}")
        print(f"  attention_mask.shape: {attention_mask.shape}")
        print(f"  labels: {labels}, dtype: {labels.dtype}")
        print(f"  input_ids range: [{input_ids.min()}, {input_ids.max()}], vocab_size: {vocab_size}")
        if input_ids.max() >= vocab_size:
            print(f"  WARNING: input_ids has values >= vocab_size!")
        print(f"  num_labels: {model.config.num_labels}")

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        print(f"  logits: {outputs.logits}")
        print(f"  logits has NaN: {torch.isnan(outputs.logits).any()}")
        sanity_loss = outputs.loss.item()
    print(f"  Sanity check loss: {sanity_loss:.4f}")
    if sanity_loss == 0.0 or np.isnan(sanity_loss):
        raise RuntimeError(
            f"Sanity check failed: loss={sanity_loss}. "
            "Check label mapping and model configuration."
        )
    model.train()

    # Training arguments — MPS-safe settings
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3 if max_steps <= 0 else 100,
        max_steps=max_steps,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=1e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        max_grad_norm=1.0,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        fp16=False,
        bf16=False,
        dataloader_pin_memory=False,
        use_mps_device=(device == "mps"),
        report_to="none",
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_split,
        eval_dataset=eval_split,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("Starting training...")
    trainer.train()

    # Final evaluation
    results = trainer.evaluate()
    print("\n=== Final Evaluation ===")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Save best model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nCheckpoint saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune DeBERTa-v3-large on WildGuardMix (MPS-safe).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", choices=["mps", "cpu"], default=None,
                        help="Force device (default: auto-detect MPS/CPU)")
    parser.add_argument("--max-steps", type=int, default=-1,
                        help="Max training steps (-1 = full 3 epochs). Use e.g. 50 for validation runs.")
    args = parser.parse_args()
    main(device_override=args.device, max_steps=args.max_steps)

"""Fine-tune DeBERTa-v3-large on WildGuardMix for binary safety classification.

Target: Mac Studio M3 Ultra (MPS) at 169.254.1.2
Dataset: allenai/wildguardmix from HuggingFace
Task: binary sequence classification (safe=0, unsafe=1)
Output: checkpoints/deberta-wildguardmix/
"""

from __future__ import annotations

import os

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


def main():
    model_name = "microsoft/deberta-v3-large"
    output_dir = "checkpoints/deberta-wildguardmix"
    device = get_device()
    print(f"Device: {device}")

    # Load dataset
    print("Loading WildGuardMix...")
    ds = load_dataset("allenai/wildguardmix", "wildguardtrain")
    train_ds = ds["train"]

    # Map labels: "safe" -> 0, anything else -> 1
    def map_labels(example):
        # WildGuardMix has 'safety_label' field
        label_field = example.get("safety_label", example.get("label", ""))
        example["label"] = 0 if label_field.lower().strip() == "safe" else 1
        example["text"] = example.get("prompt", example.get("text", ""))
        return example

    train_ds = train_ds.map(map_labels)

    # Train/eval split
    split = train_ds.train_test_split(test_size=0.1, seed=42)
    train_split = split["train"]
    eval_split = split["test"]
    print(f"Train: {len(train_split)}, Eval: {len(eval_split)}")

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

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        fp16=False,  # MPS doesn't support fp16 training
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
    main()

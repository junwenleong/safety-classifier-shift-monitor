# DeBERTa Fine-tuning on Mac Studio

## Prerequisites

On the Mac Studio (169.254.1.2):

```bash
pip install torch transformers datasets scikit-learn accelerate
```

## Run

```bash
ssh 169.254.1.2
cd /path/to/safety-classifier-shift-monitor
python scripts/finetune_deberta.py
```

## Output

Checkpoint saved to `checkpoints/deberta-wildguardmix/`.

To use the fine-tuned model with the shift monitor:

```bash
export DEBERTA_CHECKPOINT_PATH=checkpoints/deberta-wildguardmix
```

## Expected Duration

~2-3 hours on M3 Ultra with MPS acceleration (WildGuardMix ~87k examples, 3 epochs).

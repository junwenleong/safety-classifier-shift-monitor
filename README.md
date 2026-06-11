# Shift Detection Monitor

**Paper:** [arXiv:2606.11949](https://arxiv.org/abs/2606.11949)

Online monitoring system for distributional shift in deployed safety classifiers. Uses a sliding-window KS statistic on classifier scores with empirically calibrated alarm thresholds to detect when a classifier has moved out of distribution. The system architecture includes a growing-window confidence sequence (Waudby-Smith & Ramdas 2024) with time-uniform coverage guarantees and an MMD detector on classifier embeddings; the pre-registered factorial evaluation uses the KS channel. Upon detection, a conformal abstention layer adapts decision thresholds to preserve a target error rate.

## Overview

The system operates on simulated production streams across a factorial evaluation design:

- **4 classifiers**: Llama Guard 3 (8B), ShieldGemma (9B), gpt-oss-safeguard, DeBERTa-v3-large
- **5 shift conditions**: paraphrase, code-switch, adversarial suffix, compositional/long-context, temporal
- **3 ground-truth regimes**: synthetic onset (A), temporal split (B), adversarial success (C)

## Headline Findings

1. **Density-ratio collapse in high-dimensional embeddings.** Weighted conformal correction fails for generative classifiers (Llama Guard, ShieldGemma) because logistic regression achieves perfect separability in 3584–4096-d embedding space, driving all density ratios to zero. A diagnostic PCA experiment confirms this is a dimensionality artifact: projecting to ≤32 dimensions breaks the separability and restores coverage.

2. **Architectural crossover invisible to single-classifier studies.** Encoders detect paraphrase fast (28–35 steps) but adversarial suffix slow; decoders show the opposite. The classifier × shift interaction explains 18.5% of detection latency variance — monitoring must be tuned per-classifier.

3. **Cross-classifier canary architecture.** GCG suffixes optimized against DeBERTa appear anomalous to Llama Guard (scores shift *toward* unsafe). Our results suggest that a second architecturally-different classifier may serve as a distributional canary — detecting evasion attacks not by classifying content, but by flagging anomalous score distributions that the targeted classifier cannot see. (Based on 22-example corpus; larger-scale validation needed.)

## Installation

```bash
# Clone and install
git clone <repository-url>
cd shift-detection-monitor
pip install -e ".[dev]"

# For classifier inference (requires GPU)
pip install -e ".[dev,inference]"
```

### Requirements

- Python 3.11+
- NumPy, SciPy, scikit-learn, statsmodels, PyYAML, Pydantic v2
- pytest + Hypothesis (dev)
- PyTorch + HuggingFace Transformers (inference, optional)

### Reproducing Encoder Checkpoints

The two encoder classifiers (DeBERTa-v3-large and Text-Moderation) are fine-tuned on WildGuardMix. Checkpoints are not committed (too large); reproduce them with:

```bash
# DeBERTa-v3-large → checkpoints/deberta-wildguardmix/
python scripts/finetune_deberta.py

# Text-Moderation (KoalaAI) → checkpoints/text-moderation-wildguardmix/
python scripts/finetune_text_moderation.py
```

Both scripts use `allenai/wildguardmix` from HuggingFace, binary classification (safe=0, unsafe=1), with early stopping. Training takes ~2 hours per model on Apple Silicon (MPS) or NVIDIA GPU.

Set environment variables before running evaluation:
```bash
export DEBERTA_CHECKPOINT_PATH=checkpoints/deberta-wildguardmix
export TEXT_MODERATION_CHECKPOINT_PATH=checkpoints/text-moderation-wildguardmix
```

Decoder classifiers (Llama Guard 3, ShieldGemma) use their original pre-trained weights from HuggingFace and require no additional setup.

## Quickstart

### Validate a configuration

```bash
python -m shift_detection_monitor.cli validate-config --config configs/quick_test.yaml
```

### Run evaluation

```bash
python -m shift_detection_monitor.cli run --config configs/quick_test.yaml --output results/output.jsonl
```

### Build a shift dataset

```bash
python -m shift_detection_monitor.cli build-dataset \
    --shift-condition paraphrase \
    --source data/reference/source.jsonl \
    --output data/shifted/paraphrase/output.jsonl \
    --seed 42
```

## Configuration

Configuration files are YAML. Three presets are provided:

| Config | Purpose | Scale |
|--------|---------|-------|
| `configs/default.yaml` | Default values | Full factorial |
| `configs/factorial_full.yaml` | Pre-registered evaluation | 3,600 runs |
| `configs/quick_test.yaml` | Development / CI | 2 runs |

Key parameters:

- `detector.alpha`: Significance level (default 0.05)
- `detector.window_size`: Sliding window size (default 200)
- `factorial.seeds`: Random seeds for reproducibility
- `controls.n_negative_runs`: Negative control runs per classifier (default 20)

## Running Tests

```bash
# Fast tests (excludes integration)
pytest tests/ --ignore=tests/integration -q --tb=short

# Property-based tests only
pytest tests/ -k "test_" --ignore=tests/integration -q

# Integration tests (slower)
pytest tests/integration/ -q --tb=short -m slow

# Full suite
pytest tests/ -q --tb=short
```

## Architecture

```
shift_detection_monitor/
├── classifiers/          # Safety classifier adapters (Protocol-based)
├── stream/               # Stream simulation and dataset building
├── detection/            # CS engine, MMD, KS, alarm controller
├── adaptation/           # Conformal abstention, density ratio estimation
├── evaluation/           # Harness, metrics, variance decomposition
├── serialization/        # Config and result I/O (YAML/JSON/JSONL)
└── cli.py                # CLI entry point
```

### Key Components

- **ConfidenceSequenceEngine**: Betting-based CS (Waudby-Smith & Ramdas 2024) with time-uniform coverage guarantees
- **MMDDetector**: Gaussian kernel MMD on classifier embeddings with bootstrap calibration
- **KSDetector**: One-sample KS statistic on score distributions
- **AlarmController**: Multiplicity-corrected alarm management (Bonferroni/Šidák)
- **ConformalAbstentionLayer**: Split-conformal prediction sets with optional weighted correction
- **EvaluationHarness**: Full factorial orchestration with negative/positive controls
- **VarianceDecomposer**: Hierarchical ANOVA for factor importance analysis

## License

Research code. See LICENSE for details.

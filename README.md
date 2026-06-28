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

3. **Score-disagreement monitoring detects gradient-based evasion.** Any un-targeted classifier detects when the primary is under GCG attack (p<10⁻¹², n=49). The divergence is attack-specific (not generic OOD) and confidence-gated: when the canary is confident, a divergence-minimising attacker stalls at a predicted equilibrium (gap=1/(2λ), validated within 95% CI). Architecture diversity is not required for detection (η²=0.011) but provides transfer robustness (0% cross-family transfer vs 30% within-family).

4. **The Deployment Configuration Trap.** An initial screen of 35 frontier LLMs appeared to show a 'reversed scaling law' where flagship/reasoning models (o3, gpt-5) were ceiling-clipped at 1.0 on all inputs. Investigation revealed this was entirely a token-budget parsing artifact: reasoning models require `max_completion_tokens≥200` for their internal chain-of-thought; at the standard `max_tokens=16`, they produce empty responses incorrectly parsed as maximum-unsafe. With proper configuration, ALL models discriminate correctly (benign≈0.0, harmful≈0.8). Only 2 models (gpt-5-nano, gpt-5.5) exhibit genuine content-aware refusal. The optimal production choice remains `gpt-4o-mini` ($0.15/1M tokens, ≥71% guaranteed detection, <1.5% FPR at N=1000) due to cost, not capability.

5. **Calibration-free monitoring via scan martingale.** A conformal test martingale replaces empirical threshold tuning, achieving FAR≤1% uniformly across all 4 classifiers with no per-model calibration (vs 2–9.5% spread under empirical KS calibration). The martingale's value is operational simplicity (deploy once, guaranteed FAR), not superior detection power — with proper per-condition calibration, KS matches or exceeds it.

6. **Black-box optimization hits a hard floor.** 500-step coordinate-ascent attacks against `gpt-4o-mini` reveal a saturating fitness landscape: all improvements occur within the first 50 steps, after which the score is completely flat (Δ=0.000 for steps 50–500). Mean final score: 0.88 (threshold: 0.5). Increasing attacker budget provides zero marginal advantage.

7. **CoT Suffocation phase transition.** Reasoning models exhibit a sigmoid response curve: P(response|T_r) = σ(k·(T_r - T_50)), with T_50(benign)=46 tokens and T_50(adversarial)=154 tokens for o3. The 3.3× gap explains why standard API configurations (max_tokens=16) cause total failure on complex inputs while simple queries occasionally succeed.

8. **Monitorability law falsified.** The n=4 correlation (r=0.97) between null-score std and detection latency was an encoder/decoder gap artifact. Within-family evaluation (n=6 encoder variants) yields r=0.21, p=0.70 — monitorability is not a predictable intrinsic property of score-distribution geometry.

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

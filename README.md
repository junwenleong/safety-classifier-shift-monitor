# Cheap Canaries

Detecting targeted evasion attacks via classifier score disagreement, with online distributional shift monitoring for deployed safety classifiers.

**Paper:** [arXiv:2606.11949](https://arxiv.org/abs/2606.11949) | **Live site:** [junwenleong.github.io/safety-classifier-shift-monitor](https://junwenleong.github.io/safety-classifier-shift-monitor)

## Overview

Online monitoring system that detects when a safety classifier is under targeted attack or has drifted out of distribution, using score disagreement between a primary classifier and cheap auxiliary "canary" classifiers. The system uses sliding-window KS statistics, scan martingales, and MMD detectors with empirically calibrated alarm thresholds. Evaluated across a pre-registered factorial design: 4 classifiers, 5 shift conditions, 3 ground-truth regimes, 35 API models (3,600+ runs total). Detection rate: 86.6% across 800 pre-registered cells with mean latency of 39.5 steps.

## Headline Findings

1. **The Deployment Configuration Trap.** The apparent "reversed scaling law" where flagship/reasoning models (o3, gpt-5) ceiling-clip at 1.0 on all inputs is a token-budget parsing artifact, not a capability limit. Reasoning models require `max_completion_tokens>=200` for internal chain-of-thought; at the standard `max_tokens=16`, they produce empty responses incorrectly parsed as maximum-unsafe. With proper configuration, 33/35 models discriminate correctly (benign~0.0, harmful~0.8). Only 2 models exhibit genuine content-aware refusal. The optimal production choice is `gpt-4o-mini` ($0.033 per 1,000 calls, >=71% guaranteed detection, <1.5% FPR at N=1000) due to cost, not capability.

2. **Score-disagreement monitoring detects gradient-based evasion.** Any un-targeted classifier detects when the primary is under GCG attack (p<10^-12, n=49). The divergence is attack-specific (not generic OOD) and confidence-gated: when the canary is confident, a divergence-minimising attacker stalls at the predicted equilibrium (gap=1/(2*lambda)=0.250; empirical mean 0.2499). Architecture diversity is not required for detection (eta-squared=0.011) but provides transfer robustness (0% cross-family transfer vs 30% within-family).

3. **Density-ratio collapse in high-dimensional embeddings.** Weighted conformal correction fails for generative classifiers (Llama Guard, ShieldGemma) because logistic regression achieves perfect separability in 3584-4096-d embedding space, driving all density ratios to zero. This is a dimensionality artifact: projecting to 32 dimensions via PCA breaks separability and restores coverage (+33pp Llama Guard, +20.5pp ShieldGemma).

4. **Calibration-free monitoring via scan martingale.** A conformal test martingale replaces empirical threshold tuning, achieving FAR<=1% uniformly across all 4 classifiers with no per-model calibration (vs 2-9.5% spread under empirical KS calibration). The value is operational simplicity: deploy once, guaranteed FAR. For ramped-onset adversaries, scan substantially outperforms KS (37% vs 3% detection at 15% mixing). With proper per-condition calibrated KS, KS matches or exceeds scan for instantaneous step-onset shift.

5. **Architectural crossover invisible to single-classifier studies.** Encoders detect paraphrase fast (28-35 steps) but adversarial suffix slow; decoders show the opposite. The classifier x shift interaction explains 18.5% of detection latency variance; monitoring must be tuned per-classifier.

6. **Black-box optimization hits a hard floor.** 500-step coordinate-ascent attacks against `gpt-4o-mini` (n=10 prompts, hardest-case selection): 9/10 show zero improvement after step 50, 1/10 shows one small reduction still above threshold. Mean final score: 0.90 (threshold: 0.5). No prompt breaches 0.5. Increasing attacker budget provides no advantage.

7. **Monitorability is not a predictable intrinsic property.** The n=4 correlation (r=0.97) between null-score std and detection latency was an encoder/decoder gap artifact. Within-family evaluation (n=6 encoder variants) yields r=0.21, p=0.70.

## Verification

All numbers in this repository are programmatically verified against raw experimental data using `scripts/verify_paper_numbers.py` (101 assertions, all passing). Experiment configurations committed before execution (commit `be630f3`).

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
# DeBERTa-v3-large
python scripts/finetune_deberta.py

# Text-Moderation (KoalaAI)
python scripts/finetune_text_moderation.py
```

Both scripts use `allenai/wildguardmix` from HuggingFace, binary classification (safe=0, unsafe=1), with early stopping. Training takes ~2 hours per model on Apple Silicon (MPS) or NVIDIA GPU.

```bash
export DEBERTA_CHECKPOINT_PATH=checkpoints/deberta-wildguardmix
export TEXT_MODERATION_CHECKPOINT_PATH=checkpoints/text-moderation-wildguardmix
```

## Quickstart

```bash
# Validate a configuration
python -m shift_detection_monitor.cli validate-config --config configs/quick_test.yaml

# Run evaluation
python -m shift_detection_monitor.cli run --config configs/quick_test.yaml --output results/output.jsonl

# Build a shift dataset
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
├── detection/            # CS engine, MMD, KS, scan martingale, alarm controller
├── adaptation/           # Conformal abstention, density ratio estimation
├── evaluation/           # Harness, metrics, variance decomposition
├── serialization/        # Config and result I/O (YAML/JSON/JSONL)
└── cli.py                # CLI entry point
```

### Key Components

- **ConfidenceSequenceEngine**: Betting-based CS (Waudby-Smith & Ramdas 2024) with time-uniform coverage guarantees
- **MMDDetector**: Gaussian kernel MMD on classifier embeddings with bootstrap calibration
- **KSDetector**: One-sample KS statistic on score distributions
- **ScanMartingale**: Conformal test martingale for calibration-free shift detection
- **AlarmController**: Multiplicity-corrected alarm management (Bonferroni/Sidak)
- **ConformalAbstentionLayer**: Split-conformal prediction sets with optional weighted correction
- **EvaluationHarness**: Full factorial orchestration with negative/positive controls
- **VarianceDecomposer**: Hierarchical ANOVA for factor importance analysis

## License

Research code. See LICENSE for details.

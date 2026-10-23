# Cheap Canaries

Online Shift Detection and Conformal Adaptation for Deployed Safety Classifiers.

**Paper:** [arXiv:2606.11949](https://arxiv.org/abs/2606.11949) | **Live site:** [junwenleong.github.io/safety-classifier-shift-monitor](https://junwenleong.github.io/safety-classifier-shift-monitor)

## Overview

Reasoning models deployed as safety monitors suffer **reasoning-token budget starvation**: on o3, GCG-appended harmful inputs needed a 2.5x larger completion-token cap than simple benign inputs (T50_adv=117 vs T50_benign=46; harmfulness, length and suffix are not separated), so standard deployments silently fail on exactly the inputs they must catch. We characterize this within a broader adversarial threat model for deployed safety classifiers, including a local stationary-score calculation for one fixed attack objective (gap=1/(2*lambda)) whose final gaps cluster near that value at high lambda. It is not a general security boundary. The system uses score disagreement between a primary classifier and cheap auxiliary "canary" classifiers, with sliding-window KS statistics, scan martingales, and MMD detectors. Evaluated across a pre-registered factorial design (an executed 800-cell subset): 4 classifiers, 5 shift conditions, 3 ground-truth regimes, 35 API models screened. Control-validated detection rate: 86.8% across the 800 cells with mean latency of 39.7 steps at w=100 (regenerated with the corrected KS statistic; the pre-fix values were 86.6% and 39.5 steps, see the paper's appendix).

## Headline Findings

1. **Reasoning-token budget starvation.** On o3, GCG-appended harmful inputs needed a 2.5x larger completion-token cap than simple benign inputs to produce valid safety scores (sigmoid response curve: T50_adv=117 vs T50_benign=46 for o3, measured on an extended 10-300 token grid). Standard low-budget deployments (`max_completion_tokens<=16`) silently starve the monitor, producing empty responses parsed as maximum-unsafe; all three tested reasoning models return 0/100 valid scores at 16 tokens. The adversarial-versus-benign comparison is confounded (harmful prompts with GCG suffixes against short benign prompts), so the 2.5x figure is specific to o3, this prompt and this input set. The paper does not report a split of the 35 models into discriminators and refusers. A cost-effective canary candidate is `gpt-4o-mini` ($0.033 per 1,000 calls, Wilson 95% lower bound of 71% detection on the evaluated corpus, <1.5% FPR at N=1000).

2. **Score-disagreement monitoring detects gradient-based evasion.** An un-targeted Llama Guard canary flags 75.5% (37/49) of the GCG inputs that DeBERTa scores as safe after attack (the best single canary reaches 93.9%), against 12.2% for length-matched random suffixes (p<10^-12). The divergence is attack-specific, not generic OOD. For a divergence-minimising attacker the final gaps cluster near 1/(2*lambda) among blocked runs (median 0.250, 14/20 blocked at lambda=2.0), but the blocked frequency does not replicate (70% primary pair, 30% replication pair, 50% in the lambda sweep) and the paper withdraws the earlier claim that the stall is confidence-gated. Architectural diversity gave no clear advantage in this four-model ensemble (eta-squared=0.011, p=0.070). The defence is not absolute: about a quarter of the inputs DeBERTa scores as safe also pass the Llama Guard canary, and same-family joint optimisation reaches both-below-0.5 on 11/20 prompts (6/13 among prompts both models initially flagged).

3. **Density-ratio collapse in high-dimensional embeddings.** Weighted conformal correction fails for 3 of 4 tested classifiers because logistic regression achieves perfect separability in 768-4096-d embedding space, driving all density ratios to zero. Only the DeBERTa-v3-base primary classifier (768-d) avoids collapse. Projecting to 32 dimensions via PCA breaks separability and restores coverage on temporal shift (+33pp Llama Guard, +20.5pp ShieldGemma). The coverage experiments use safe-only calibration data and are a diagnostic of the estimator, not a clean covariate-shift guarantee.

4. **Calibration-free monitoring via scan martingale.** A conformal test martingale replaces empirical threshold tuning, giving FAR<=1% on the null streams tested for all 4 classifiers with no per-model calibration (vs 2-9.5% spread under empirical KS calibration). This is an empirical result, not a proven anytime guarantee. For ramped-onset adversaries the scan detected 60% / 93% of streams at 15% / 20% mixing against 33-57% / 67-87% for KS, depending on the KS threshold rule. With a per-condition calibrated KS, KS matches or exceeds scan for instantaneous step-onset shift.

5. **Architectural crossover invisible to single-classifier studies.** Encoders detect paraphrase fast (28-35 steps) but adversarial suffix slow; decoders show the opposite. The classifier x shift interaction explains 18.6% of detection latency variance; monitoring must be tuned per-classifier.

6. **Black-box optimisation hit a floor in this budget.** 500-step coordinate-ascent attacks against `gpt-4o-mini` (10 runs on at most 9 distinct high-baseline prompts): 9/10 show zero improvement after step 50, 1/10 shows one small reduction still above threshold. Mean final score: 0.90 (threshold: 0.5). No run breaches 0.5, which at n=10 bounds the breach rate only to about 31%.

7. **Monitorability is not a predictable intrinsic property.** The n=4 correlation (r=0.97) between null-score std and detection latency was an encoder/decoder gap artefact. Within-family evaluation (n=6 encoder variants) yields r=0.21, p=0.70.

8. **CUSUM/EWMA close the sub-threshold blind spot (v5).** Persistent low-rate contamination (4-12% mixing) that evades both KS and scan martingale is detectable by long-memory sequential detectors. CUSUM (k=0.5) and EWMA (lambda=0.05) detect down to 4% mixing at 90% detection rate (MDM=4%), at the cost of increased latency (~930 steps vs ~365 for KS at higher mixing). At 12% mixing (the previously identified blind-spot boundary), CUSUM detects at 88 steps with 100% power.

9. **KS is sensitive to benign heterogeneity (v5).** Under production-like traffic drawn from multiple benign sources (code, creative writing, chat), KS FAR inflates from 4.3% to 23.4% (5.4x) because it detects any distributional difference, not only adversarial shift. Recalibrating KS on representative production traffic recovers controlled FAR (2.6%). Scored with the primary DeBERTa-v3-base classifier on 3,768 prompts from 5 sources (`results/v5_heterogeneous_real/`). **CUSUM's behaviour under this same heterogeneous traffic is not established.** On that run CUSUM saturates (FAR near 100%), while a separately labelled run of the same nominal experiment (`results/v5_heterogeneous/`) shows CUSUM essentially unchanged; the two point in opposite directions and were not reconciled, so we make no robustness (or non-robustness) claim for CUSUM on heterogeneous traffic.

## Verification

Two earlier results are withdrawn and not part of the findings above: a canary-detection case study using a locally run open-weight model (no committed artefact), and a post-hoc validation of newer frontier models (the update script scored the same truncated prompt in both arms, so it did not measure false positives). See FINDINGS.md's Reproducibility Notes.

## Verification

`scripts/verify_paper_numbers.py` checks 225 selected reported values against committed result files (all passing as of 2026-10-20). It does not re-run experiments or validate methodology. `make reproduce` runs the sub-threshold experiment and this verifier; it does not regenerate every table and figure. Experiment configurations were committed before execution (commit `0380167`; see `HISTORY_REWRITE.md` for the mapping from earlier identifiers).

Known limitations of this repository: the shifted corpora under `data/shifted/` are git-ignored in the public repository, encoder checkpoints are not committed, and `paper/latex/smsr_connection.tex` is a withdrawn appendix that is not part of the paper.

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

The two encoder classifiers (DeBERTa-v3-base and Text-Moderation) are fine-tuned on WildGuardMix. Checkpoints are not committed (too large); reproduce them with:

```bash
# DeBERTa-v3-base
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

- **ConfidenceSequenceEngine**: Betting-based CS (Waudby-Smith & Ramdas 2024); the time-uniform guarantee holds only with a known reference mean, and the paper's factorial estimates it in-sample
- **MMDDetector**: Gaussian kernel MMD on classifier embeddings with bootstrap calibration
- **KSDetector**: One-sample KS statistic on score distributions
- **ScanMartingale**: Conformal test martingale for shift detection with shared parameters across classifiers (empirical FAR, no proven anytime bound)
- **AlarmController**: Multiplicity-corrected alarm management (Bonferroni/Sidak)
- **ConformalAbstentionLayer**: Split-conformal prediction sets with optional weighted correction
- **EvaluationHarness**: Full factorial orchestration with negative/positive controls
- **VarianceDecomposer**: Hierarchical ANOVA for factor importance analysis

## License

Source code: MIT (`LICENSE`). Data, figures and documentation: CC BY 4.0 (`LICENSE-DATA`).
Preregistrations are frozen, cited records and are not edited. Please cite arXiv:2606.11949.

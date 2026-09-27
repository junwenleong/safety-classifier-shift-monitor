"""v5 Experiment 3: Heterogeneous negative controls.

Tests whether false alarm rates inflate under production-like benign traffic
variance compared to homogeneous WildGuardMix controls. If yes, recalibrates.

Pre-registration: docs/pre_registration_v5.md, Experiment 3.

** DATA-CONSISTENCY WARNING (added after audit, verified by deterministic
re-run with seed=42 against the committed data): ** three result files
exist for this experiment (results/v5_heterogeneous/heterogeneous_far_*.json
and results/v5_heterogeneous_real/heterogeneous_far_results.json) and they
do not agree. paper.tex cites "KS FAR inflates from 4.3% to 23.4%... CUSUM
is unaffected (FAR 6.1% -> 6.6%)" as if from one experiment. Re-running the
exact detector code in this file against
results/v5_heterogeneous_real/heterogeneous_scores.json (the file whose KS
numbers match the paper exactly: homo=4.3%, hetero=23.4%, recal=2.6%) gives
CUSUM homo=100%, hetero=100%, recal=100% -- completely saturated, the
opposite of "unaffected". EWMA on the same data gives homo=88.8%,
hetero=11.2% -- also badly miscalibrated (and inverted: lower on
heterogeneous than homogeneous). The cited "CUSUM 6.1%->6.6%" figure
instead matches results/v5_heterogeneous/heterogeneous_far_results.json (no
"_real"), a DIFFERENT run against a different homogeneous reference array,
not the one that produced the cited KS numbers. Running this exact script
today (RESULTS_DIR is hardcoded to "results/v5_heterogeneous") would not
reproduce the paper's cited KS figures either. Root cause not fully
resolved: results/null_scores.json's current "deberta" array contains a
few extreme outliers (~0.99) that appear to be inflating CUSUM's mu0/sigma
calibration when bootstrap-resampled into a 500-item reference; whether
that array was different at the time the "6.1%->6.6%" run happened is
unknown. Do not cite the CUSUM/EWMA heterogeneous-robustness numbers from
this paragraph without re-deriving them from a single, internally
consistent run.

Usage:
    # Full experiment (requires Mac Studio for DeBERTa scoring)
    .venv/bin/python scripts/exp_heterogeneous_controls.py

    # Score-only mode (just score the benign corpus, save scores)
    .venv/bin/python scripts/exp_heterogeneous_controls.py --score-only

    # Analyse-only mode (use pre-scored results)
    .venv/bin/python scripts/exp_heterogeneous_controls.py --analyse-only
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS_DIR = Path("results/v5_heterogeneous")
SCORES_FILE = RESULTS_DIR / "heterogeneous_scores.json"

# Pre-registration targets
TARGET_PROMPTS = 5000
N_STREAMS = 1000
STREAM_LENGTH = 1000
FAR_TARGET = 0.05


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_benign_corpus(n_target: int = TARGET_PROMPTS) -> dict[str, list[str]]:
    """Build multi-source benign corpus per pre-registration.

    Sources:
    - WildGuardMix unharmful (40%)
    - OpenAssistant/Dolly (20%)
    - Code/technical (15%)
    - Creative/general (15%)
    - Multi-turn fragments (10%)
    """
    from datasets import load_dataset

    corpus = {}
    rng = np.random.default_rng(42)

    # Source 1: WildGuardMix unharmful (40%)
    n_wgm = int(n_target * 0.40)
    log(f"Loading WildGuardMix unharmful (target: {n_wgm})...")
    try:
        ds = load_dataset("allenai/wildguardmix", "wildguardtrain", split="train")
        wgm_safe = [ex["prompt"] for ex in ds
                    if ex["prompt_harm_label"] == "unharmful" and len(ex["prompt"]) > 20]
        indices = rng.permutation(len(wgm_safe))[:n_wgm]
        corpus["wildguardmix"] = [wgm_safe[i] for i in indices]
        log(f"  WildGuardMix: {len(corpus['wildguardmix'])} prompts")
    except Exception as e:
        log(f"  WildGuardMix failed: {e}")
        corpus["wildguardmix"] = []

    # Source 2: OpenAssistant (20%)
    n_oa = int(n_target * 0.20)
    log(f"Loading OpenAssistant (target: {n_oa})...")
    try:
        ds = load_dataset("OpenAssistant/oasst1", split="train")
        # Take human messages (role=prompter) as benign prompts
        oa_prompts = [ex["text"] for ex in ds
                      if ex.get("role") == "prompter" and len(ex["text"]) > 20]
        if not oa_prompts:
            # Fallback: just take all texts > 20 chars
            oa_prompts = [ex["text"] for ex in ds if len(ex["text"]) > 20]
        indices = rng.permutation(len(oa_prompts))[:n_oa]
        corpus["openassistant"] = [oa_prompts[i] for i in indices]
        log(f"  OpenAssistant: {len(corpus['openassistant'])} prompts")
    except Exception as e:
        log(f"  OpenAssistant failed: {e}, trying Dolly...")
        try:
            ds = load_dataset("databricks/databricks-dolly-15k", split="train")
            dolly = [ex["instruction"] for ex in ds if len(ex["instruction"]) > 20]
            indices = rng.permutation(len(dolly))[:n_oa]
            corpus["openassistant"] = [dolly[i] for i in indices]
            log(f"  Dolly fallback: {len(corpus['openassistant'])} prompts")
        except Exception as e2:
            log(f"  Dolly also failed: {e2}")
            corpus["openassistant"] = []

    # Source 3: Code/technical (15%)
    n_code = int(n_target * 0.15)
    log(f"Loading code/technical (target: {n_code})...")
    try:
        # Use HumanEval-style coding prompts
        ds = load_dataset("openai/openai_humaneval", split="test")
        code_prompts = [ex["prompt"] for ex in ds if len(ex["prompt"]) > 20]
        # Supplement with more if needed
        if len(code_prompts) < n_code:
            ds2 = load_dataset("codeparrot/apps", split="train", streaming=True)
            for ex in ds2:
                if len(code_prompts) >= n_code:
                    break
                q = ex.get("question", "")
                if len(q) > 20:
                    code_prompts.append(q)
    except Exception:
        code_prompts = []

    if not code_prompts:
        # Generate synthetic technical prompts as fallback
        templates = [
            "Write a function that {}",
            "How do I implement {} in Python?",
            "Explain the algorithm for {}",
            "Debug this code: {}",
            "What is the time complexity of {}?",
        ]
        topics = ["binary search", "merge sort", "BFS", "dynamic programming",
                  "hash tables", "linked lists", "tree traversal", "regex parsing",
                  "file I/O", "socket programming", "JSON parsing", "CSV processing",
                  "database queries", "API authentication", "rate limiting"]
        code_prompts = [t.format(topic) for t in templates for topic in topics]

    indices = rng.permutation(len(code_prompts))[:n_code]
    corpus["code_technical"] = [code_prompts[i] for i in indices]
    log(f"  Code/technical: {len(corpus['code_technical'])} prompts")

    # Source 4: Creative/general (15%)
    n_creative = int(n_target * 0.15)
    log(f"Loading creative/general (target: {n_creative})...")
    try:
        ds = load_dataset("Anthropic/hh-rlhf", split="train", streaming=True)
        creative = []
        for ex in ds:
            if len(creative) >= n_creative * 3:
                break
            text = ex.get("chosen", "")
            # Extract human turns
            if "\n\nHuman:" in text:
                turns = text.split("\n\nHuman:")
                for turn in turns[1:]:
                    prompt = turn.split("\n\nAssistant:")[0].strip()
                    if len(prompt) > 20:
                        creative.append(prompt)
    except Exception:
        creative = []

    if len(creative) < n_creative:
        # Fallback: generic creative prompts
        creative_templates = [
            "Write a short story about {}",
            "Describe a day in the life of {}",
            "What would happen if {}?",
            "Compare and contrast {} and {}",
            "Summarize the main ideas of {}",
        ]
        nouns = ["a scientist", "a teacher", "ancient Rome", "the ocean",
                 "the future", "a small town", "mathematics", "music",
                 "artificial intelligence", "space exploration"]
        for t in creative_templates:
            for n1 in nouns:
                if "{}" in t and t.count("{}") == 1:
                    creative.append(t.format(n1))
                elif t.count("{}") == 2:
                    for n2 in nouns:
                        if n1 != n2:
                            creative.append(t.format(n1, n2))

    indices = rng.permutation(len(creative))[:n_creative]
    corpus["creative_general"] = [creative[i] for i in indices]
    log(f"  Creative/general: {len(corpus['creative_general'])} prompts")

    # Source 5: Multi-turn fragments (10%)
    n_multi = int(n_target * 0.10)
    log(f"Loading multi-turn fragments (target: {n_multi})...")
    # Simulate multi-turn by concatenating consecutive prompts
    all_available = []
    for src in ["wildguardmix", "openassistant"]:
        all_available.extend(corpus.get(src, []))

    multi = []
    if len(all_available) >= 2 * n_multi:
        for i in range(0, min(2 * n_multi, len(all_available) - 1), 2):
            multi.append(f"Previous: {all_available[i][:100]}\nFollow-up: {all_available[i+1][:100]}")
    else:
        multi = all_available[:n_multi]

    corpus["multi_turn"] = multi[:n_multi]
    log(f"  Multi-turn: {len(corpus['multi_turn'])} prompts")

    total = sum(len(v) for v in corpus.values())
    log(f"\nTotal corpus: {total} prompts across {len(corpus)} sources")
    return corpus


def score_corpus(corpus: dict[str, list[str]], checkpoint_path: str) -> dict[str, list[float]]:
    """Score all prompts with DeBERTa. Requires GPU/MPS."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    log(f"Scoring on device: {device}")

    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint_path, num_labels=2).to(device)
    model.eval()

    scores_by_source = {}
    total_prompts = sum(len(v) for v in corpus.values())
    done = 0

    for source, prompts in corpus.items():
        source_scores = []
        batch_size = 32

        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", truncation=True,
                             max_length=512, padding=True).to(device)
            with torch.no_grad():
                logits = model(**inputs).logits
                probs = torch.softmax(logits, dim=1)[:, 1]  # P(unsafe)
                source_scores.extend(probs.cpu().numpy().tolist())

            done += len(batch)
            if done % 500 == 0:
                log(f"  Scored {done}/{total_prompts}...")

        scores_by_source[source] = source_scores
        log(f"  {source}: {len(source_scores)} scored, "
            f"mean={np.mean(source_scores):.4f}, "
            f"frac>0.5={np.mean(np.array(source_scores) > 0.5):.3f}")

    return scores_by_source


def filter_benign(scores_by_source: dict[str, list[float]],
                  corpus: dict[str, list[str]]) -> tuple[np.ndarray, dict]:
    """Remove prompts scored > 0.5 (false benign). Return clean scores."""
    clean_scores = []
    stats = {}

    for source in scores_by_source:
        src_scores = np.array(scores_by_source[source])
        mask = src_scores <= 0.5
        n_removed = int((~mask).sum())
        clean = src_scores[mask]
        clean_scores.extend(clean.tolist())
        stats[source] = {
            "total": len(src_scores),
            "removed": n_removed,
            "kept": int(mask.sum()),
            "mean_score": float(np.mean(clean)) if len(clean) > 0 else None,
        }
        if n_removed > 0:
            log(f"  Filtered {source}: removed {n_removed}/{len(src_scores)} "
                f"(scored > 0.5)")

    return np.array(clean_scores), stats


def run_far_analysis(heterogeneous_scores: np.ndarray,
                     homogeneous_scores: np.ndarray) -> dict:
    """Run FAR inflation analysis: heterogeneous vs homogeneous null streams."""
    rng = np.random.default_rng(42)

    results = {"detectors": {}}

    for detector_name, detector_fn in [
        ("ks_w100", lambda stream, ref: _ks_detector(stream, ref, window=100)),
        ("cusum_k0.5", lambda stream, ref: _cusum_detector(stream, ref, k=0.5)),
        ("ewma_l0.05", lambda stream, ref: _ewma_detector(stream, ref, lam=0.05)),
    ]:
        # Homogeneous null streams
        homo_alarms = 0
        for _ in range(N_STREAMS):
            stream = rng.choice(homogeneous_scores, size=STREAM_LENGTH, replace=True)
            ref = rng.choice(homogeneous_scores, size=500, replace=True)
            if detector_fn(stream, ref):
                homo_alarms += 1
        homo_far = homo_alarms / N_STREAMS

        # Heterogeneous null streams
        hetero_alarms = 0
        for _ in range(N_STREAMS):
            stream = rng.choice(heterogeneous_scores, size=STREAM_LENGTH, replace=True)
            ref = rng.choice(homogeneous_scores, size=500, replace=True)  # calibrated on homo
            if detector_fn(stream, ref):
                hetero_alarms += 1
        hetero_far = hetero_alarms / N_STREAMS

        # Recalibrated: use heterogeneous reference
        recal_alarms = 0
        for _ in range(N_STREAMS):
            stream = rng.choice(heterogeneous_scores, size=STREAM_LENGTH, replace=True)
            ref = rng.choice(heterogeneous_scores, size=500, replace=True)
            if detector_fn(stream, ref):
                recal_alarms += 1
        recal_far = recal_alarms / N_STREAMS

        results["detectors"][detector_name] = {
            "homogeneous_far": round(homo_far, 4),
            "heterogeneous_far": round(hetero_far, 4),
            "recalibrated_far": round(recal_far, 4),
            "inflation": round(hetero_far - homo_far, 4),
            "recalibration_recovers": recal_far <= FAR_TARGET,
        }
        log(f"  {detector_name}: homo={homo_far:.3f}, hetero={hetero_far:.3f}, "
            f"recal={recal_far:.3f}, inflation={hetero_far - homo_far:+.3f}")

    return results


def _ks_detector(stream: np.ndarray, ref: np.ndarray, window: int = 100) -> bool:
    """Sliding-window KS detector."""
    from scipy.stats import ks_2samp
    threshold = 0.20  # Calibrated from null scores
    for start in range(0, len(stream) - window, window // 2):
        stat, _ = ks_2samp(ref, stream[start:start + window])
        if stat > threshold:
            return True
    return False


def _cusum_detector(stream: np.ndarray, ref: np.ndarray, k: float = 0.5) -> bool:
    """CUSUM detector (upper)."""
    mu0 = np.mean(ref)
    sigma = np.std(ref) + 1e-8
    h = 5.0 * sigma  # Threshold
    s_pos = 0.0
    for x in stream:
        s_pos = max(0, s_pos + (x - mu0) / sigma - k)
        if s_pos > h:
            return True
    return False


def _ewma_detector(stream: np.ndarray, ref: np.ndarray, lam: float = 0.05) -> bool:
    """EWMA detector."""
    mu0 = np.mean(ref)
    sigma = np.std(ref) + 1e-8
    L = 3.0  # Control limit in sigma units
    z = mu0
    for i, x in enumerate(stream):
        z = lam * x + (1 - lam) * z
        sigma_z = sigma * np.sqrt(lam / (2 - lam) * (1 - (1 - lam) ** (2 * (i + 1))))
        if abs(z - mu0) > L * sigma_z:
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description="v5 Heterogeneous controls experiment")
    parser.add_argument("--score-only", action="store_true",
                        help="Only score corpus, don't run FAR analysis")
    parser.add_argument("--analyse-only", action="store_true",
                        help="Only run FAR analysis on pre-scored data")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/deberta-wildguardmix",
                        help="DeBERTa checkpoint path")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not args.analyse_only:
        # Build corpus
        log("Building heterogeneous benign corpus...")
        corpus = build_benign_corpus()

        # Score with DeBERTa
        log("\nScoring with DeBERTa...")
        scores_by_source = score_corpus(corpus, args.checkpoint)

        # Filter out false-benign
        log("\nFiltering false-benign prompts (score > 0.5)...")
        clean_scores, filter_stats = filter_benign(scores_by_source, corpus)
        log(f"Clean corpus: {len(clean_scores)} prompts")

        # Save scores
        save_data = {
            "scores_by_source": {k: [round(s, 6) for s in v]
                                for k, v in scores_by_source.items()},
            "filter_stats": filter_stats,
            "clean_scores": [round(s, 6) for s in clean_scores.tolist()],
            "n_total": len(clean_scores),
        }
        with open(SCORES_FILE, "w") as f:
            json.dump(save_data, f, indent=2)
        log(f"Saved scores to {SCORES_FILE}")

        if args.score_only:
            return

    else:
        # Load pre-scored
        log("Loading pre-scored data...")
        with open(SCORES_FILE) as f:
            save_data = json.load(f)
        clean_scores = np.array(save_data["clean_scores"])
        log(f"Loaded {len(clean_scores)} scored prompts")

    # Load homogeneous baseline (from factorial null scores)
    null_scores_file = Path("results/null_scores.json")
    if null_scores_file.exists():
        with open(null_scores_file) as f:
            null_data = json.load(f)
        homo_scores = np.array(null_data.get("deberta", []))
        log(f"Loaded {len(homo_scores)} homogeneous null scores")
    else:
        log("ERROR: results/null_scores.json not found. Cannot compare.")
        sys.exit(1)

    # Run FAR analysis
    log(f"\nRunning FAR analysis ({N_STREAMS} streams × {STREAM_LENGTH} length)...")
    far_results = run_far_analysis(clean_scores, homo_scores)

    # Source-conditional FAR
    log("\nSource-conditional FAR analysis...")
    source_far = {}
    scores_by_source = save_data.get("scores_by_source", {}) if not args.analyse_only else \
        json.load(open(SCORES_FILE)).get("scores_by_source", {})

    rng = np.random.default_rng(123)
    for source, src_scores_list in scores_by_source.items():
        src_scores = np.array([s for s in src_scores_list if s <= 0.5])
        if len(src_scores) < 100:
            source_far[source] = {"n": len(src_scores), "far": None, "note": "too few"}
            continue
        alarms = 0
        n_test = min(200, N_STREAMS)
        for _ in range(n_test):
            stream = rng.choice(src_scores, size=min(STREAM_LENGTH, len(src_scores)), replace=True)
            ref = rng.choice(homo_scores, size=500, replace=True)
            if _ks_detector(stream, ref):
                alarms += 1
        source_far[source] = {
            "n": len(src_scores),
            "mean_score": round(float(np.mean(src_scores)), 4),
            "std_score": round(float(np.std(src_scores)), 4),
            "far": round(alarms / n_test, 4),
        }
        log(f"  {source}: FAR={alarms/n_test:.3f} (n={len(src_scores)}, "
            f"mean={np.mean(src_scores):.4f})")

    far_results["source_conditional"] = source_far

    # Save final results
    output_file = RESULTS_DIR / "heterogeneous_far_results.json"
    with open(output_file, "w") as f:
        json.dump(far_results, f, indent=2)
    log(f"\nSaved results to {output_file}")

    # Summary
    log(f"\n{'='*60}")
    log("HETEROGENEOUS CONTROLS SUMMARY")
    log(f"{'='*60}")
    for det, vals in far_results["detectors"].items():
        log(f"  {det}:")
        log(f"    Homogeneous FAR:   {vals['homogeneous_far']:.3f}")
        log(f"    Heterogeneous FAR: {vals['heterogeneous_far']:.3f} "
            f"(inflation: {vals['inflation']:+.3f})")
        log(f"    Recalibrated FAR:  {vals['recalibrated_far']:.3f} "
            f"({'✅ recovers' if vals['recalibration_recovers'] else '❌ still inflated'})")


if __name__ == "__main__":
    main()

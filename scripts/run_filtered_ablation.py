"""Step 6: Filtered paraphrase ablation — remove refusals, compare detection latency.

Filters LLM refusals from the paraphrase corpus and reruns detection on the
clean subset. Compares latency to unfiltered results.

Usage:
    .venv/bin/python scripts/run_filtered_ablation.py
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from shift_detection_monitor.detection.ks_detector import KSDetector
from shift_detection_monitor.detection.reference_window import ReferenceWindow
from shift_detection_monitor.types import StreamRecord

PARAPHRASE_PATH = Path("data/shifted/paraphrase/output.jsonl")
OUTPUT = Path("results/filtered_ablation_results.json")

CLASSIFIERS = ["deberta", "llama-guard"]
SEEDS = list(range(5))
N_REFERENCE = 500
SHIFT_ONSET = 500
WINDOW_SIZE = 100
N_CALIBRATION = 50
CAL_PCT = 97

# Refusal detection heuristics
REFUSAL_PATTERNS = [
    "i cannot", "i can't", "i'm not able", "i am not able",
    "i apologize", "sorry, but", "i'm sorry",
    "as an ai", "as a language model", "as an artificial",
    "i must decline", "i won't", "i will not",
    "it's not appropriate", "not appropriate for me",
    "i don't think i should", "i shouldn't",
    "harmful content", "cannot assist", "unable to help",
    "against my guidelines", "violates my",
]


def is_refusal(text: str) -> bool:
    """Heuristic refusal detection."""
    text_lower = text.lower()[:500]
    return any(pat in text_lower for pat in REFUSAL_PATTERNS)


def load_paraphrase_corpus():
    """Load and split into clean vs refusal."""
    with open(PARAPHRASE_PATH) as f:
        raw = [json.loads(line) for line in f if line.strip()]

    for r in raw:
        if "shifted" in r and "text" not in r:
            r["text"] = r["shifted"]
        if len(r.get("text", "")) > 1000:
            r["text"] = r["text"][:1000]

    clean = [r for r in raw if not is_refusal(r.get("text", ""))]
    refusals = [r for r in raw if is_refusal(r.get("text", ""))]

    return clean, refusals, raw


def get_classifier(name: str):
    if name == "deberta":
        from shift_detection_monitor.classifiers.deberta import DeBERTaAdapter
        return DeBERTaAdapter()
    elif name == "llama-guard":
        from shift_detection_monitor.classifiers.llama_guard import LlamaGuard3Adapter
        return LlamaGuard3Adapter()


def load_reference():
    from datasets import load_dataset
    ds = load_dataset("allenai/wildguardmix", "wildguardtrain", split="train")
    ds = ds.filter(lambda x: x["prompt_harm_label"] == "unharmful")
    ds = ds.shuffle(seed=42)
    prompts = ds["prompt"]
    return [{"text": prompts[i]} for i in range(N_REFERENCE * 3)]


def run_detection(classifier, reference, shifted, seed, threshold):
    """Run KS detection on a stream."""
    rng = random.Random(seed)
    ref_pool = list(reference[:N_REFERENCE])
    shift_pool = list(shifted[:300])
    rng.shuffle(ref_pool)
    rng.shuffle(shift_pool)

    ref_window = ReferenceWindow(min_size=WINDOW_SIZE, n_bootstrap=100)
    records = []

    # Reference phase
    for i in range(min(WINDOW_SIZE, len(ref_pool))):
        out = classifier.predict(ref_pool[i]["text"])
        rec = StreamRecord(time_step=i, text=ref_pool[i]["text"], score=out.score,
                          representation=out.representation, ground_truth_label=None,
                          is_shifted=False, source_dataset="reference", shift_condition=None)
        ref_window.add(rec)
        records.append(rec)

    frozen = ref_window.freeze()
    ks_det = KSDetector(frozen_stats=frozen, window_size=WINDOW_SIZE)
    for rec in records:
        ks_det.update(rec)

    # Continue reference + shifted
    alarm_step = None
    step = WINDOW_SIZE
    for i in range(WINDOW_SIZE, len(ref_pool)):
        if step >= SHIFT_ONSET:
            break
        out = classifier.predict(ref_pool[i]["text"])
        rec = StreamRecord(time_step=step, text=ref_pool[i]["text"], score=out.score,
                          representation=out.representation, ground_truth_label=None,
                          is_shifted=False, source_dataset="reference", shift_condition=None)
        val = ks_det.update(rec)
        if val > threshold and step > 2 * WINDOW_SIZE and alarm_step is None:
            alarm_step = step
        step += 1

    # Shifted phase
    for i in range(len(shift_pool)):
        out = classifier.predict(shift_pool[i]["text"])
        rec = StreamRecord(time_step=step, text=shift_pool[i]["text"], score=out.score,
                          representation=out.representation, ground_truth_label=None,
                          is_shifted=True, source_dataset="shifted", shift_condition="paraphrase")
        val = ks_det.update(rec)
        if val > threshold and step > 2 * WINDOW_SIZE and alarm_step is None:
            alarm_step = step
        step += 1

    latency = (alarm_step - SHIFT_ONSET) if alarm_step is not None else None
    return {"alarm_step": alarm_step, "detection_latency": latency}


def calibrate_threshold(classifier, reference):
    """Calibrate from null streams."""
    from scripts.run_gradual_drift import calibrate_threshold as _cal
    neg_pool = reference[N_REFERENCE:]
    return _cal(classifier, reference[:N_REFERENCE], neg_pool)


def main():
    wall_start = time.time()
    clean, refusals, full = load_paraphrase_corpus()
    print("=" * 60)
    print("FILTERED PARAPHRASE ABLATION")
    print(f"  Full corpus: {len(full)} examples")
    print(f"  Clean (non-refusal): {len(clean)} examples ({len(clean)/len(full)*100:.1f}%)")
    print(f"  Refusals filtered: {len(refusals)} examples ({len(refusals)/len(full)*100:.1f}%)")
    print("=" * 60)

    reference = load_reference()
    results = {"corpus_stats": {"total": len(full), "clean": len(clean), "refusals": len(refusals)}}

    for clf_name in CLASSIFIERS:
        print(f"\n  Classifier: {clf_name}")
        classifier = get_classifier(clf_name)
        threshold = calibrate_threshold(classifier, reference)
        print(f"    Threshold: {threshold:.4f}")

        unfiltered_lats = []
        filtered_lats = []

        for seed in SEEDS:
            # Unfiltered (original corpus)
            res_full = run_detection(classifier, reference, full[:300], seed, threshold)
            unfiltered_lats.append(res_full["detection_latency"])

            # Filtered (refusals removed)
            res_clean = run_detection(classifier, reference, clean[:300], seed, threshold)
            filtered_lats.append(res_clean["detection_latency"])

            print(f"    Seed {seed}: unfiltered={res_full['detection_latency']}, filtered={res_clean['detection_latency']}")

        valid_unf = [l for l in unfiltered_lats if l is not None and l >= 0]
        valid_filt = [l for l in filtered_lats if l is not None and l >= 0]

        results[clf_name] = {
            "unfiltered": {"latencies": unfiltered_lats, "mean": float(np.mean(valid_unf)) if valid_unf else None, "detect_rate": len(valid_unf) / len(SEEDS)},
            "filtered": {"latencies": filtered_lats, "mean": float(np.mean(valid_filt)) if valid_filt else None, "detect_rate": len(valid_filt) / len(SEEDS)},
        }
        print(f"    Unfiltered: mean={np.mean(valid_unf):.1f} ({len(valid_unf)}/{len(SEEDS)} detected)" if valid_unf else "    Unfiltered: no detections")
        print(f"    Filtered:   mean={np.mean(valid_filt):.1f} ({len(valid_filt)}/{len(SEEDS)} detected)" if valid_filt else "    Filtered: no detections")

        del classifier

    with open(OUTPUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Wall time: {(time.time()-wall_start)/60:.1f} min")
    print(f"  Saved to {OUTPUT}")


if __name__ == "__main__":
    main()

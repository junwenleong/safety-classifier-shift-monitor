"""Step 2: Extract and cache embeddings + scores for all Mac Studio experiments.

Caches per-step scores and embeddings as .npz files. Unblocks:
- CS growing-window evaluation (needs scores)
- MMD evaluation (needs embeddings)
- PCA conformal (needs embeddings)
- Embedding displacement scatter (needs embeddings)

Usage:
    .venv/bin/python scripts/cache_embeddings.py
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_full_canary import SHIFT_CORPORA, load_shifted

CLASSIFIERS = ["deberta", "text-moderation", "llama-guard", "shieldgemma"]
SHIFTS = ["paraphrase", "temporal", "adversarial-suffix"]
SEEDS = list(range(10))
N_REFERENCE = 500
SHIFT_ONSET = 500
CACHE_DIR = Path("results/cached_streams")


def get_classifier(name: str):
    if name == "deberta":
        from shift_detection_monitor.classifiers.deberta import DeBERTaAdapter
        return DeBERTaAdapter()
    elif name == "text-moderation":
        from shift_detection_monitor.classifiers.gpt_oss_safeguard import TextModerationAdapter
        return TextModerationAdapter()
    elif name == "shieldgemma":
        from shift_detection_monitor.classifiers.shieldgemma import ShieldGemmaAdapter
        return ShieldGemmaAdapter()
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


def run_stream(classifier, reference, shifted, seed):
    """Run a full stream, return arrays of scores and embeddings."""
    rng = random.Random(seed)
    ref_pool = list(reference[:N_REFERENCE])
    shift_pool = list(shifted)
    rng.shuffle(ref_pool)
    rng.shuffle(shift_pool)

    scores = []
    embeddings = []
    is_shifted_flags = []

    # Reference phase (steps 0 to SHIFT_ONSET-1)
    for i in range(min(SHIFT_ONSET, len(ref_pool))):
        out = classifier.predict(ref_pool[i]["text"])
        scores.append(out.score)
        embeddings.append(out.representation)
        is_shifted_flags.append(False)

    # Shifted phase (steps SHIFT_ONSET onward, mixing=1.0)
    for i in range(len(shift_pool)):
        out = classifier.predict(shift_pool[i]["text"])
        scores.append(out.score)
        embeddings.append(out.representation)
        is_shifted_flags.append(True)

    scores_arr = np.array(scores, dtype=np.float64)
    is_shifted_arr = np.array(is_shifted_flags, dtype=bool)

    # Embeddings may be None for some classifiers
    if embeddings[0] is not None:
        embed_arr = np.array(embeddings, dtype=np.float32)
    else:
        embed_arr = None

    return scores_arr, embed_arr, is_shifted_arr


def main():
    wall_start = time.time()
    print("=" * 60)
    print("EMBEDDING CACHING: 4 classifiers × 3 shifts × 10 seeds")
    print("=" * 60)

    reference = load_reference()
    print(f"Reference pool: {len(reference)} examples")

    for clf_name in CLASSIFIERS:
        print(f"\n{'='*60}")
        print(f"Classifier: {clf_name}")
        print(f"{'='*60}")
        classifier = get_classifier(clf_name)

        for shift in SHIFTS:
            shifted = load_shifted(SHIFT_CORPORA.get(shift))
            print(f"\n  Shift: {shift} ({len(shifted)} examples)")

            for seed in SEEDS:
                out_dir = CACHE_DIR / clf_name / shift
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"seed_{seed}.npz"

                if out_path.exists():
                    print(f"    Seed {seed}: cached, skipping")
                    continue

                t0 = time.time()
                scores, embeddings, is_shifted = run_stream(
                    classifier, reference, shifted, seed
                )
                elapsed = time.time() - t0

                save_dict = {
                    "scores": scores,
                    "is_shifted": is_shifted,
                }
                if embeddings is not None:
                    save_dict["embeddings"] = embeddings

                np.savez_compressed(out_path, **save_dict)
                print(f"    Seed {seed}: {len(scores)} steps, {elapsed:.1f}s")

        # Free GPU memory between classifiers
        del classifier

    wall_time = time.time() - wall_start
    print(f"\n{'='*60}")
    print(f"Done. Total wall time: {wall_time/3600:.1f} hours")
    print(f"Cache dir: {CACHE_DIR}")


if __name__ == "__main__":
    main()

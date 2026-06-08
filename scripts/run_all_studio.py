"""Master runner: execute all Mac Studio experiments in order.

Usage:
    export DEBERTA_CHECKPOINT_PATH=checkpoints/deberta-wildguardmix
    export TEXT_MODERATION_CHECKPOINT_PATH=checkpoints/text-moderation-wildguardmix
    .venv/bin/python scripts/run_all_studio.py
"""

import subprocess
import sys
import time
from pathlib import Path

SCRIPTS = [
    ("Step 1: Gradual-drift experiment", "scripts/run_gradual_drift.py"),
    ("Step 2: Cache embeddings (GPU-heavy)", "scripts/cache_embeddings.py"),
    ("Step 3: Embedding displacement scatter", "scripts/embedding_displacement.py"),
    ("Step 4: CS growing-window evaluation", "scripts/run_cs_evaluation.py"),
    ("Step 5: PCA conformal sweep", "scripts/run_pca_conformal_sweep.py"),
    ("Step 6: Filtered paraphrase ablation", "scripts/run_filtered_ablation.py"),
    ("Step 7: MMD evaluation", "scripts/run_mmd_evaluation.py"),
]

PYTHON = Path(".venv/bin/python")


def main():
    print("=" * 70)
    print("MAC STUDIO — FULL EXPERIMENT RUNNER")
    print("=" * 70)

    total_start = time.time()
    results = []

    for i, (name, script) in enumerate(SCRIPTS, 1):
        print(f"\n{'='*70}")
        print(f"  [{i}/{len(SCRIPTS)}] {name}")
        print(f"  Script: {script}")
        print(f"{'='*70}\n")

        t0 = time.time()
        try:
            proc = subprocess.run(
                [str(PYTHON), script],
                cwd=str(Path(__file__).parent.parent),
                capture_output=False,
            )
            elapsed = time.time() - t0
            status = "✓" if proc.returncode == 0 else f"✗ (exit {proc.returncode})"
        except Exception as e:
            elapsed = time.time() - t0
            status = f"✗ ({e})"

        results.append((name, status, elapsed))
        print(f"\n  → {status} ({elapsed/60:.1f} min)")

        if proc.returncode != 0:
            print(f"\n  ⚠ {name} failed. Continue? (y/n) ", end="", flush=True)
            if input().strip().lower() != "y":
                break

    # Summary
    total_time = time.time() - total_start
    print(f"\n\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for name, status, elapsed in results:
        print(f"  {status} {name} ({elapsed/60:.1f} min)")
    print(f"\n  Total wall time: {total_time/3600:.1f} hours")


if __name__ == "__main__":
    main()

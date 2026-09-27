"""EXPLORATORY (post-hoc, v5 audit): No-shift source-vs-source control for the
density-ratio collapse / PCA-recovery claim.

Question: the paper reports that weighted conformal recovery collapses at native
embedding dimension (ESS ~= n_cal, all weights floored) and 'recovers' after PCA
to <=32 dims. Is this collapse driven by *shift*, or is it an interpolation
artefact of n << d (a logistic classifier can perfectly separate ANY two random
subsets of points in high dimension)?

Design (matches the paper's exact DRE code path):
  - Embed IN-DISTRIBUTION (source) data ONLY. No shifted data at all.
  - Randomly split embeddings into pseudo-source (n=300) and pseudo-target (n=200).
  - Fit DensityRatioEstimator (logistic, sklearn LogisticRegression, solver=lbfgs,
    C=1.0, max_iter=1000), clip weights to [1/C, C] with C=10 -- identical to
    shift_detection_monitor/adaptation/density_ratio.py.
  - Report: train accuracy, held-out AUC (5-fold on the source-vs-target logistic
    task), weight histogram / fraction at floor, ESS at native dim and after PCA
    to 8/16/32/64/128.
  - 8 random splits (>=5 required).

If the no-shift control ALSO collapses at native d and recovers at 32, the paper's
collapse is an n<<d interpolation artefact, not shift-driven.

Encoders run locally on 24GB (DeBERTa 1024-d fine-tuned checkpoint; Text-Moderation
768-d base). Llama Guard / ShieldGemma decoders are handled elsewhere / pending.

Output: results/noshift_dre_control.json
Usage:
    DEBERTA_CHECKPOINT_PATH=checkpoints/deberta-wildguardmix \\
      .venv/bin/python scripts/exp_noshift_dre_control.py
"""
from __future__ import annotations
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

RESULTS_DIR = Path("results")
SOURCE_FILE = Path("data/reference/source.jsonl")
EMB_CACHE = RESULTS_DIR / "noshift_source_embeddings.npz"
OUTPUT_FILE = RESULTS_DIR / "noshift_dre_control.json"

N_PSEUDO_SOURCE = 300
N_PSEUDO_TARGET = 200
N_SPLITS = 8
PCA_DIMS = [8, 16, 32, 64, 128]
MAX_WEIGHT = 10.0  # clip C, identical to DensityRatioEstimator default

# DRE hyperparameters, identical to shift_detection_monitor/adaptation/density_ratio.py
DRE_PENALTY = "l2"
DRE_C = 1.0
DRE_SOLVER = "lbfgs"
DRE_MAX_ITER = 1000


def load_source_texts(n_needed):
    texts = []
    with open(SOURCE_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            texts.append(json.loads(line)["text"])
    return texts


def embed_all(texts, classifier_name):
    """Embed all texts with the given classifier's penultimate [CLS] representation."""
    if classifier_name == "deberta":
        from shift_detection_monitor.classifiers.deberta import DeBERTaAdapter
        clf = DeBERTaAdapter()
    elif classifier_name == "text-moderation":
        from shift_detection_monitor.classifiers.gpt_oss_safeguard import TextModerationAdapter
        clf = TextModerationAdapter()
    else:
        raise ValueError(classifier_name)
    embs = []
    for i, t in enumerate(texts):
        out = clf.predict(t)
        embs.append(out.representation)
        if (i + 1) % 100 == 0:
            print(f"    {classifier_name}: embedded {i+1}/{len(texts)}")
    return np.array(embs, dtype=np.float64)


def dre_fit_weights(source_emb, target_emb):
    """Replicate DensityRatioEstimator._fit_logistic + weights() exactly.

    Returns (weights_on_source_calibration, train_accuracy, model, X, y).
    """
    X = np.vstack([source_emb, target_emb])
    y = np.concatenate([np.zeros(len(source_emb)), np.ones(len(target_emb))])
    model = LogisticRegression(max_iter=DRE_MAX_ITER, solver=DRE_SOLVER,
                               C=DRE_C, penalty=DRE_PENALTY)
    model.fit(X, y)
    train_acc = model.score(X, y)
    # weights() on the source (calibration) points: exp(decision), clipped.
    decision = model.decision_function(source_emb)
    max_log = np.log(MAX_WEIGHT)
    clipped = np.clip(decision, -max_log, max_log)
    ratios = np.exp(clipped)
    ratios = np.clip(ratios, 1.0 / MAX_WEIGHT, MAX_WEIGHT)
    return ratios, float(train_acc), model, X, y


def ess(weights):
    return float((weights.sum() ** 2) / (weights ** 2).sum())


def frac_at_floor(weights, tol=1e-6):
    return float(np.mean(weights <= (1.0 / MAX_WEIGHT) + tol))


def frac_at_ceil(weights, tol=1e-6):
    return float(np.mean(weights >= MAX_WEIGHT - tol))


def heldout_auc(X, y):
    """5-fold stratified held-out AUC of the source-vs-target logistic task."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    aucs = []
    for tr, te in skf.split(X, y):
        m = LogisticRegression(max_iter=DRE_MAX_ITER, solver=DRE_SOLVER,
                               C=DRE_C, penalty=DRE_PENALTY)
        m.fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        # guard against a fold with one class (shouldn't happen with stratify)
        if len(np.unique(y[te])) < 2:
            continue
        aucs.append(roc_auc_score(y[te], p))
    return float(np.mean(aucs)) if aucs else None


def run_control_for_classifier(embeddings, name, rng):
    native_d = embeddings.shape[1]
    n_total = embeddings.shape[0]
    per_split = []
    for s in range(N_SPLITS):
        idx = rng.permutation(n_total)
        src_idx = idx[:N_PSEUDO_SOURCE]
        tgt_idx = idx[N_PSEUDO_SOURCE:N_PSEUDO_SOURCE + N_PSEUDO_TARGET]
        src = embeddings[src_idx]
        tgt = embeddings[tgt_idx]

        # Native dimension
        w_native, acc_native, _, X, y = dre_fit_weights(src, tgt)
        auc_native = heldout_auc(X, y)
        native = {
            "dim": native_d,
            "train_accuracy": acc_native,
            "heldout_auc": auc_native,
            "ess": ess(w_native),
            "frac_at_floor": frac_at_floor(w_native),
            "frac_at_ceil": frac_at_ceil(w_native),
            "max_weight": float(w_native.max()),
            "min_weight": float(w_native.min()),
            "weight_hist": np.histogram(w_native, bins=10, range=(0.0, MAX_WEIGHT))[0].tolist(),
        }

        # PCA-reduced. Fit PCA on the COMBINED source+target (same as
        # run_pca_conformal_sweep.estimate_density_ratios), then DRE on reduced.
        pca_results = {}
        combined = np.vstack([src, tgt])
        for dim in PCA_DIMS:
            if dim >= native_d:
                continue
            pca = PCA(n_components=dim)
            combined_pca = pca.fit_transform(combined)
            var_ret = float(pca.explained_variance_ratio_.sum())
            src_pca = combined_pca[:len(src)]
            tgt_pca = combined_pca[len(src):]
            w_p, acc_p, _, Xp, yp = dre_fit_weights(src_pca, tgt_pca)
            auc_p = heldout_auc(Xp, yp)
            pca_results[str(dim)] = {
                "train_accuracy": acc_p,
                "heldout_auc": auc_p,
                "ess": ess(w_p),
                "frac_at_floor": frac_at_floor(w_p),
                "variance_retained": var_ret,
                "weight_hist": np.histogram(w_p, bins=10, range=(0.0, MAX_WEIGHT))[0].tolist(),
            }
        per_split.append({"split": s, "native": native, "pca": pca_results})

    # Aggregate across splits
    def agg(getter):
        vals = [getter(sp) for sp in per_split if getter(sp) is not None]
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                "min": float(np.min(vals)), "max": float(np.max(vals))} if vals else None

    summary = {
        "native_dim": native_d,
        "native": {
            "train_accuracy": agg(lambda sp: sp["native"]["train_accuracy"]),
            "heldout_auc": agg(lambda sp: sp["native"]["heldout_auc"]),
            "ess": agg(lambda sp: sp["native"]["ess"]),
            "frac_at_floor": agg(lambda sp: sp["native"]["frac_at_floor"]),
        },
        "pca": {},
    }
    for dim in PCA_DIMS:
        if dim >= native_d:
            continue
        d = str(dim)
        summary["pca"][d] = {
            "train_accuracy": agg(lambda sp, d=d: sp["pca"][d]["train_accuracy"]),
            "heldout_auc": agg(lambda sp, d=d: sp["pca"][d]["heldout_auc"]),
            "ess": agg(lambda sp, d=d: sp["pca"][d]["ess"]),
            "frac_at_floor": agg(lambda sp, d=d: sp["pca"][d]["frac_at_floor"]),
            "variance_retained": agg(lambda sp, d=d: sp["pca"][d]["variance_retained"]),
        }
    return {"summary": summary, "per_split": per_split}


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(20260927)

    # Enough texts for the largest split usage (source+target from disjoint pool
    # is 500; we sample 300+200=500 disjoint from the 500-example source pool).
    texts = load_source_texts(N_PSEUDO_SOURCE + N_PSEUDO_TARGET)
    print(f"Loaded {len(texts)} in-distribution source texts.")

    classifiers = ["deberta", "text-moderation"]

    # Embed (cache to avoid recompute).
    emb_store = {}
    if EMB_CACHE.exists():
        cached = np.load(EMB_CACHE, allow_pickle=True)
        for name in classifiers:
            key = name.replace("-", "_")
            if key in cached:
                emb_store[name] = cached[key]
                print(f"Loaded cached embeddings for {name}: {emb_store[name].shape}")

    to_embed = [c for c in classifiers if c not in emb_store]
    for name in to_embed:
        print(f"Embedding with {name} ...")
        if name == "deberta":
            os.environ.setdefault("DEBERTA_CHECKPOINT_PATH", "checkpoints/deberta-wildguardmix")
        emb_store[name] = embed_all(texts, name)
        print(f"  {name} embeddings: {emb_store[name].shape}")

    # Save cache
    np.savez(EMB_CACHE, **{name.replace("-", "_"): emb for name, emb in emb_store.items()})

    out = {
        "_meta": {
            "label": "EXPLORATORY (post-hoc, v5 audit)",
            "purpose": "No-shift source-vs-source control for density-ratio collapse / PCA recovery.",
            "n_pseudo_source": N_PSEUDO_SOURCE,
            "n_pseudo_target": N_PSEUDO_TARGET,
            "n_splits": N_SPLITS,
            "pca_dims": PCA_DIMS,
            "source_file": str(SOURCE_FILE),
            "n_source_texts": len(texts),
            "dre_hyperparameters": {
                "penalty": DRE_PENALTY,
                "C": DRE_C,
                "solver": DRE_SOLVER,
                "max_iter": DRE_MAX_ITER,
                "weight_clip_C": MAX_WEIGHT,
                "standardisation": "none (raw penultimate [CLS] embeddings, matching paper code path)",
                "pca_fit_set": "combined pseudo-source + pseudo-target (matches run_pca_conformal_sweep.estimate_density_ratios)",
            },
            "run_utc": datetime.now(timezone.utc).isoformat(),
        },
        "classifiers": {},
    }

    for name in classifiers:
        print(f"\n=== No-shift control: {name} (d={emb_store[name].shape[1]}) ===")
        res = run_control_for_classifier(emb_store[name], name, rng)
        out["classifiers"][name] = res
        s = res["summary"]
        print(f"  native d={s['native_dim']}: train_acc={s['native']['train_accuracy']['mean']:.3f} "
              f"AUC={s['native']['heldout_auc']['mean']:.3f} "
              f"ESS={s['native']['ess']['mean']:.1f} "
              f"frac_floor={s['native']['frac_at_floor']['mean']:.3f}")
        for dim in PCA_DIMS:
            if str(dim) in s["pca"]:
                p = s["pca"][str(dim)]
                print(f"  PCA {dim:>3}: train_acc={p['train_accuracy']['mean']:.3f} "
                      f"AUC={p['heldout_auc']['mean']:.3f} "
                      f"ESS={p['ess']['mean']:.1f} frac_floor={p['frac_at_floor']['mean']:.3f} "
                      f"var_ret={p['variance_retained']['mean']:.3f}")

    # Note the pending decoders explicitly.
    out["_meta"]["pending_on_mac_studio"] = (
        "Llama-Guard-3-8B and ShieldGemma no-shift controls: decoder embeddings not "
        "run on this 24GB machine (Llama Guard tested for feasibility separately; "
        "ShieldGemma not cached). Pending on Mac Studio."
    )

    OUTPUT_FILE.write_text(json.dumps(out, indent=2))
    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

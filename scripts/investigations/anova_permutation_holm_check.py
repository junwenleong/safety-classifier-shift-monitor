"""Check the paper's ANOVA permutation p-values and the Holm-adjusted p-values.

Two things the paper states without a stored generator:
  (a) "1,000 permutations for ANOVA ... all p < 0.001" (Table tab:anova);
  (b) "all 8 highlighted comparisons survive Holm-Bonferroni (weakest adjusted p = 0.044)".

This script (numpy only, no inference) reports, for the original factorial rows
and for the corrected-KS regeneration:

ANOVA (sequential sums of squares, order classifier -> shift -> interaction,
valid detections only; this is how the paper's eta-squared are defined):
  1. observed eta-squared;
  2. the repository's permutation test (scripts/run_variance_decomposition.py:
     labels of classifier and shift permuted jointly against latency, 1,000
     permutations, p = (count+1)/(B+1)), which cannot return less than 1/1001;
  3. Freedman-Lane permutation tests of each term given the terms before it
     (10,000 permutations), which test the interaction itself;
  4. the same on stream-level means (the two window sizes of one stream averaged),
     because the w=100 and w=200 cells of a stream share the same data.

Holm family (the 8 comparisons of scripts/bca_holm_bonferroni.py): raw p-values
by large-sample permutation (200,000), with p = (count+1)/(B+1), at the cell
level (as in the paper) and at the stream level (window sizes averaged).

Usage: REPRO_MANIFEST_DISABLE=1 .venv/bin/python scripts/investigations/anova_permutation_holm_check.py
"""

from __future__ import annotations

import json
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)  # checked via rss() instead
R = Path(__file__).resolve().parents[2]
FILES = {
    "original": R / "results/factorial_results.jsonl",
    "committed_ks_regeneration": R / "results/factorial_regeneration/factorial_results_committed_ks.jsonl",
    "corrected_ks_regeneration": R / "results/factorial_regeneration/factorial_results_corrected_ks.jsonl",
}
OUT = R / "results/anova_permutation_holm_check.json"
B_ANOVA = 10_000
B_HOLM = 200_000
CLFS = ["deberta", "text-moderation", "llama-guard", "shieldgemma"]
ENC, DEC = ["deberta", "text-moderation"], ["llama-guard", "shieldgemma"]


def load(path):
    rows = [json.loads(line) for line in open(path) if line.strip()]
    for r in rows:
        r["valid"] = (r["detection_latency"] is not None and r["detection_latency"] >= 0
                      and r["neg_clean"] is True)
    return rows


def stream_level(rows):
    """Average the valid windows of each (classifier, shift, seed) stream."""
    g = defaultdict(list)
    for r in rows:
        if r["valid"]:
            g[(r["classifier"], r["shift_condition"], r["seed"])].append(r["detection_latency"])
    return [{"classifier": k[0], "shift_condition": k[1], "seed": k[2], "y": float(np.mean(v))}
            for k, v in g.items()]


def dummies(a):
    lv = sorted(set(a))
    return [(a == x).astype(float) for x in lv[1:]]


def design(C, S):
    dc, ds = dummies(C), dummies(S)
    di = [a * b for a in dc for b in ds]
    one = [np.ones(len(C))]
    return one, dc, ds, di


NONFINITE = 0


def rss(y, cols):
    """Residual sum of squares. Non-finite results are counted and abort the run:
    numpy on macOS Accelerate can print spurious 'invalid value in matmul'
    warnings, and a NaN would silently count as 'not as extreme' in a p-value."""
    global NONFINITE
    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fit = X @ beta
    res = y - fit
    val = float(res @ res)
    if not np.isfinite(val):
        NONFINITE += 1
        raise FloatingPointError("non-finite residual sum of squares")
    return val, fit, res


def anova(y, C, S, rng, b):
    one, dc, ds, di = design(C, S)
    m0 = one
    m1 = one + dc
    m2 = one + dc + ds
    m3 = one + dc + ds + di
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r0, _, _ = rss(y, m0)
    r1, _, _ = rss(y, m1)
    r2, _, _ = rss(y, m2)
    r3, _, _ = rss(y, m3)
    obs = {"classifier": (r0 - r1) / ss_tot, "shift": (r1 - r2) / ss_tot,
           "interaction": (r2 - r3) / ss_tot, "residual": r3 / ss_tot}
    n = len(y)
    df = {"classifier": len(dc), "shift": len(ds), "interaction": len(di), "resid": n - len(m3)}
    fstat = {"classifier": ((r0 - r1) / df["classifier"]) / (r3 / df["resid"]),
             "shift": ((r1 - r2) / df["shift"]) / (r3 / df["resid"]),
             "interaction": ((r2 - r3) / df["interaction"]) / (r3 / df["resid"])}

    # (2) repository test: joint permutation of the label columns, 1,000 permutations
    rng2 = np.random.default_rng(42)
    cnt = {"classifier": 0, "shift": 0, "interaction": 0}
    for _ in range(1000):
        p = rng2.permutation(n)
        o, a, b_, c = design(C[p], S[p])
        s0, _, _ = rss(y, o)
        s1, _, _ = rss(y, o + a)
        s2, _, _ = rss(y, o + a + b_)
        s3, _, _ = rss(y, o + a + b_ + c)
        for k, v in (("classifier", (s0 - s1) / ss_tot), ("shift", (s1 - s2) / ss_tot),
                     ("interaction", (s2 - s3) / ss_tot)):
            if v >= obs[k]:
                cnt[k] += 1
    repo_p = {k: (v + 1) / 1001 for k, v in cnt.items()}

    # (3) Freedman-Lane: permute residuals of the reduced model, refit, compare the
    #     extra-sum-of-squares F for the term (denominator: residual of the full model)
    def fl(reduced, full, fobs, dfn):
        _, fit_r, res_r = rss(y, reduced)
        c = 0
        for _ in range(b):
            ys = fit_r + rng.permutation(res_r)
            rr, _, _ = rss(ys, reduced)
            rf, _, _ = rss(ys, full)
            f = ((rr - rf) / dfn) / (rf / (n - len(full)))
            c += f >= fobs
        return (c + 1) / (b + 1)

    # the F for the interaction uses the interaction model's own residual, so the
    # observed statistic is recomputed with the same denominator convention
    def fobs_for(reduced, full, dfn):
        rr, _, _ = rss(y, reduced)
        rf, _, _ = rss(y, full)
        return ((rr - rf) / dfn) / (rf / (n - len(full)))

    fl_p = {
        "classifier": fl(m0, m1, fobs_for(m0, m1, df["classifier"]), df["classifier"]),
        "shift": fl(m1, m2, fobs_for(m1, m2, df["shift"]), df["shift"]),
        "interaction": fl(m2, m3, fobs_for(m2, m3, df["interaction"]), df["interaction"]),
    }
    return {"n": n, "eta_sq": obs, "F_full_model_denominator": fstat, "df": df,
            "repo_joint_permutation_p_1000": repo_p,
            "freedman_lane_p": fl_p, "freedman_lane_B": b}


def perm_p_unpaired(a, b, rng, B):
    comb = np.concatenate([a, b])
    n1 = len(a)
    obs = abs(a.mean() - b.mean())
    cnt = 0
    for _ in range(B):
        p = rng.permutation(comb)
        cnt += abs(p[:n1].mean() - p[n1:].mean()) >= obs
    return (cnt + 1) / (B + 1)


def perm_p_paired(d, rng, B):
    obs = abs(d.mean())
    cnt = 0
    chunk = 20_000
    for _ in range(B // chunk):
        signs = rng.choice([-1.0, 1.0], size=(chunk, len(d)))
        cnt += int((np.abs((signs * d).mean(axis=1)) >= obs).sum())
    done = (B // chunk) * chunk
    return (cnt + 1) / (done + 1)


def holm(ps):
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i][1])
    adj, prev = {}, 0.0
    for rank, i in enumerate(order):
        a = max(min(ps[i][1] * (m - rank), 1.0), prev)
        adj[ps[i][0]] = a
        prev = a
    return adj


def comparisons(rows, level, rng, B):
    valid = [r for r in rows if r["valid"]]
    w100 = {(r["classifier"], r["shift_condition"], r["seed"]): r["detection_latency"]
            for r in valid if r["window_size"] == 100}
    w200 = {(r["classifier"], r["shift_condition"], r["seed"]): r["detection_latency"]
            for r in valid if r["window_size"] == 200}
    keys = sorted(set(w100) & set(w200))
    paired = np.array([w100[k] - w200[k] for k in keys], float)
    if level == "cell":
        def get(clf=None, shift=None):
            return np.array([r["detection_latency"] for r in valid
                             if (clf is None or r["classifier"] == clf)
                             and (shift is None or r["shift_condition"] == shift)], float)
        dirty = {c: np.array([0.0 if r["neg_clean"] else 1.0 for r in rows if r["classifier"] == c])
                 for c in CLFS}
    else:
        sl = stream_level(rows)

        def get(clf=None, shift=None):
            return np.array([s["y"] for s in sl
                             if (clf is None or s["classifier"] == clf)
                             and (shift is None or s["shift_condition"] == shift)], float)
        g = defaultdict(list)
        for r in rows:
            g[(r["classifier"], r["seed"], r["shift_condition"])].append(0.0 if r["neg_clean"] else 1.0)
        dirty = {c: np.array([np.mean(v) for k, v in g.items() if k[0] == c]) for c in CLFS}

    def cat(names, shift):
        return np.concatenate([get(n, shift) for n in names])

    out = [
        ("1 decoder vs encoder, paraphrase",
         perm_p_unpaired(cat(ENC, "paraphrase"), cat(DEC, "paraphrase"), rng, B)),
        ("2 encoder vs decoder, adversarial suffix",
         perm_p_unpaired(cat(ENC, "adversarial-suffix"), cat(DEC, "adversarial-suffix"), rng, B)),
        ("3 DeBERTa adversarial vs paraphrase",
         perm_p_unpaired(get("deberta", "paraphrase"), get("deberta", "adversarial-suffix"), rng, B)),
        ("4 Llama Guard paraphrase vs adversarial",
         perm_p_unpaired(get("llama-guard", "adversarial-suffix"), get("llama-guard", "paraphrase"), rng, B)),
        ("5 window 100 vs 200 (paired)", perm_p_paired(paired, rng, B)),
        ("6 Llama Guard x code-switch vs grand mean",
         perm_p_unpaired(get("llama-guard", "code-switch"), get(), rng, B)),
        ("7 ShieldGemma paraphrase vs adversarial",
         perm_p_unpaired(get("shieldgemma", "adversarial-suffix"), get("shieldgemma", "paraphrase"), rng, B)),
        ("8 FAR DeBERTa vs Text-Moderation",
         perm_p_unpaired(dirty["deberta"], dirty["text-moderation"], rng, B)),
    ]
    adj = holm(out)
    return [{"comparison": n, "raw_p": p, "holm_adjusted_p": adj[n]} for n, p in out]


def main():
    result = {}
    for name, path in FILES.items():
        rows = load(path)
        valid = [r for r in rows if r["valid"]]
        res = {"n_cells": len(rows), "n_valid": len(valid)}
        rng = np.random.default_rng(2026)
        y = np.array([r["detection_latency"] for r in valid], float)
        C = np.array([r["classifier"] for r in valid])
        S = np.array([r["shift_condition"] for r in valid])
        res["anova_cell_level"] = anova(y, C, S, rng, B_ANOVA)
        sl = stream_level(rows)
        res["anova_stream_level"] = anova(np.array([s["y"] for s in sl]),
                                          np.array([s["classifier"] for s in sl]),
                                          np.array([s["shift_condition"] for s in sl]), rng, B_ANOVA)
        res["holm_cell_level"] = comparisons(rows, "cell", rng, B_HOLM)
        res["holm_stream_level"] = comparisons(rows, "stream", rng, B_HOLM)
        result[name] = res
        print(f"== {name}: {len(rows)} cells, {len(valid)} valid ==")
        for lvl in ("anova_cell_level", "anova_stream_level"):
            a = res[lvl]
            print(f" {lvl} n={a['n']} eta2 " + ", ".join(f"{k} {v:.3f}" for k, v in a["eta_sq"].items()))
            print(f"   repo joint-permutation p (B=1000): {a['repo_joint_permutation_p_1000']}")
            print(f"   Freedman-Lane p (B={a['freedman_lane_B']}): {a['freedman_lane_p']}")
        for lvl in ("holm_cell_level", "holm_stream_level"):
            print(f" {lvl}")
            for c in res[lvl]:
                print(f"   {c['comparison']:<45} raw {c['raw_p']:.5f}  Holm {c['holm_adjusted_p']:.5f}")
    result["non_finite_rss_events"] = NONFINITE
    print("non-finite RSS events:", NONFINITE)
    OUT.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()

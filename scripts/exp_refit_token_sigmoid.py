"""EXPLORATORY (v5 audit): Refit the o3 token-starvation sigmoid from RAW data
with bootstrap CIs and goodness-of-fit.

Two modes:
  (default)  audit the original 8-point grid (10-80, n=10) only.
  --merged   merge the original grid with the extended measured sweep
             (results/token_sweep_o3_extended.json, budgets 80-300, n=20/class)
             and refit BOTH benign and adversarial sigmoids on MEASURED points
             only, with parametric bootstrap CIs on T50 and k plus R^2.

The extended sweep makes the adversarial curve identifiable for the first time
(the original grid had 1/80 valid adversarial responses). Errors (HTTP/API) in
the extended sweep are excluded from denominators, never scored.

Reads only cached results (zero API calls). git_dirty expected (commits forbidden).
"""
from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit

RESULTS = Path("results")
rng = np.random.default_rng(42)


def sigmoid(x, k, t50):
    return 1.0 / (1.0 + np.exp(-k * (x - t50)))


def load_grid():
    data = json.loads((RESULTS / "token_sweep_o3.json").read_text())
    tokens, ben_valid, ben_n, adv_valid, adv_n = [], [], [], [], []
    for k in sorted(data.keys(), key=int):
        t = int(k)
        b = data[k]["benign_results"]
        a = data[k]["adv_results"]
        nb = sum(1 for r in b if not r["empty"])
        na = sum(1 for r in a if not r["empty"])
        tokens.append(t)
        ben_valid.append(nb); ben_n.append(len(b))
        adv_valid.append(na); adv_n.append(len(a))
    return (np.array(tokens, float), np.array(ben_valid), np.array(ben_n),
            np.array(adv_valid), np.array(adv_n))


def fit_rate(tokens, valid, n, p0):
    rate = valid / n
    try:
        popt, _ = curve_fit(sigmoid, tokens, rate, p0=p0, maxfev=20000,
                            bounds=([1e-4, 0], [5, 500]))
        pred = sigmoid(tokens, *popt)
        ss_res = np.sum((rate - pred) ** 2)
        ss_tot = np.sum((rate - rate.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        return popt, r2
    except Exception as e:
        return None, str(e)


def bootstrap_fit(tokens, valid, n, p0, B=2000):
    """Parametric bootstrap: resample binomial valid counts per budget."""
    k_s, t50_s = [], []
    for _ in range(B):
        vb = rng.binomial(n, valid / n)
        popt, _ = fit_rate(tokens, vb, n, p0)
        if popt is not None:
            k_s.append(popt[0]); t50_s.append(popt[1])
    def ci(a):
        a = np.array(a)
        return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)),
                float(np.median(a)))
    return ci(k_s), ci(t50_s), len(k_s)


def main():
    tokens, bv, bn, av, an = load_grid()
    print("RAW GRID (token_sweep_o3.json):")
    print(f"  budgets: {tokens.tolist()}")
    print(f"  benign valid/n: {list(zip(bv.tolist(), bn.tolist()))}")
    print(f"  adv valid/n:    {list(zip(av.tolist(), an.tolist()))}")
    print(f"  adv max response rate in grid: {(av/an).max():.2f} at "
          f"{tokens[(av/an).argmax()]:.0f} tokens (n valid points>0: {(av>0).sum()})")

    print("\nBENIGN fit (raw grid only, no synthetic points):")
    popt_b, r2_b = fit_rate(tokens, bv, bn, [0.1, 50])
    if popt_b is not None:
        (k_lo, k_hi, k_md), (t_lo, t_hi, t_md), nboot = bootstrap_fit(tokens, bv, bn, [0.1, 50])
        print(f"  k={popt_b[0]:.3f}  T50={popt_b[1]:.1f}  R^2={r2_b:.3f}")
        print(f"  bootstrap k 95%CI [{k_lo:.3f},{k_hi:.3f}] median {k_md:.3f}")
        print(f"  bootstrap T50 95%CI [{t_lo:.1f},{t_hi:.1f}] median {t_md:.1f}  (nboot={nboot})")

    print("\nADVERSARIAL fit (raw grid only, NO hardcoded 200-token point):")
    popt_a, r2_a = fit_rate(tokens, av, an, [0.03, 150])
    if popt_a is not None:
        print(f"  k={popt_a[0]:.3f}  T50={popt_a[1]:.1f}  R^2={r2_a}")
        (k_lo, k_hi, k_md), (t_lo, t_hi, t_md), nboot = bootstrap_fit(tokens, av, an, [0.03, 150])
        print(f"  bootstrap k 95%CI [{k_lo:.3f},{k_hi:.3f}] median {k_md:.3f}")
        print(f"  bootstrap T50 95%CI [{t_lo:.1f},{t_hi:.1f}] median {t_md:.1f}  (nboot={nboot})")
    else:
        print(f"  FIT FAILED: {r2_a}")

    # Reproduce the paper's method: add the hardcoded synthetic point
    print("\nADVERSARIAL fit REPRODUCING depth_analysis.py (adds synthetic (200, 0.80)):")
    t2 = np.append(tokens, 200.0)
    av2 = np.append(av, 8); an2 = np.append(an, 10)  # 0.80 => 8/10
    mask = (av2 / an2) > 0
    popt_a2, r2_a2 = fit_rate(t2[mask], av2[mask], an2[mask], [0.05, 100])
    if popt_a2 is not None:
        print(f"  k={popt_a2[0]:.3f}  T50={popt_a2[1]:.1f}  R^2={r2_a2}")
        print("  (paper reports k=0.030, T50=154)")

    # Cross-check reasoning_effort_sweep for adv response at higher budgets
    re = json.loads((RESULTS / "reasoning_effort_sweep.json").read_text())
    lo = re["o3"]["low"]
    adv_resp = 1 - lo["adv_empty"] / len(lo["adv_scores"])
    print(f"\nreasoning_effort_sweep o3/low: adv response rate = {adv_resp:.2f} "
          f"(adv_empty={lo['adv_empty']}/{len(lo['adv_scores'])}) at that budget")
    print("  -> contradicts the hardcoded 0.80 adv response at 200 tokens")


# ---------------------------------------------------------------------------
# MERGED MODE: original grid (10-80, n=10) + extended sweep (80-300, n=20)
# ---------------------------------------------------------------------------
def load_merged():
    """Merge the original grid with the extended measured sweep on valid-response
    RATE (valid / scored-denominator). Budgets present in both (80) are summed.
    Returns dict budget -> (benign_valid, benign_n, adv_valid, adv_n)."""
    grid = json.loads((RESULTS / "token_sweep_o3.json").read_text())
    ext = json.loads((RESULTS / "token_sweep_o3_extended.json").read_text())

    merged: dict[int, list[int]] = {}

    for k in sorted(grid.keys(), key=int):
        t = int(k)
        b = grid[k]["benign_results"]
        a = grid[k]["adv_results"]
        nb = sum(1 for r in b if not r["empty"])
        na = sum(1 for r in a if not r["empty"])
        merged[t] = [nb, len(b), na, len(a)]

    for k, v in ext["budgets"].items():
        t = int(k)
        bs = v["benign_summary"]; as_ = v["adv_summary"]
        bv, bn = bs["n_valid"], bs["n_scored_denominator"]
        av, an = as_["n_valid"], as_["n_scored_denominator"]
        if t in merged:
            merged[t][0] += bv; merged[t][1] += bn
            merged[t][2] += av; merged[t][3] += an
        else:
            merged[t] = [bv, bn, av, an]

    tokens = np.array(sorted(merged.keys()), float)
    bv = np.array([merged[int(t)][0] for t in tokens])
    bn = np.array([merged[int(t)][1] for t in tokens])
    av = np.array([merged[int(t)][2] for t in tokens])
    an = np.array([merged[int(t)][3] for t in tokens])
    return tokens, bv, bn, av, an


def main_merged():
    tokens, bv, bn, av, an = load_merged()
    print("MERGED GRID (token_sweep_o3.json + token_sweep_o3_extended.json):")
    print(f"  budgets: {tokens.tolist()}")
    print(f"  benign valid/n: {list(zip(bv.tolist(), bn.tolist()))}")
    print(f"  adv valid/n:    {list(zip(av.tolist(), an.tolist()))}")
    print(f"  adv rate:       {[round(x,3) for x in (av/an).tolist()]}")

    print("\nBENIGN fit (merged measured points, no synthetic):")
    popt_b, r2_b = fit_rate(tokens, bv, bn, [0.1, 50])
    result = {"label": "EXPLORATORY (post-hoc, v5 audit)",
              "source": ["token_sweep_o3.json", "token_sweep_o3_extended.json"]}
    if popt_b is not None:
        (k_lo, k_hi, k_md), (t_lo, t_hi, t_md), nb_ = bootstrap_fit(tokens, bv, bn, [0.1, 50])
        print(f"  k={popt_b[0]:.3f}  T50={popt_b[1]:.1f}  R^2={r2_b:.3f}")
        print(f"  bootstrap k 95%CI [{k_lo:.3f},{k_hi:.3f}] median {k_md:.3f}")
        print(f"  bootstrap T50 95%CI [{t_lo:.1f},{t_hi:.1f}] median {t_md:.1f} (nboot={nb_})")
        result["benign"] = {"k": float(popt_b[0]), "t50": float(popt_b[1]), "r2": float(r2_b),
                            "k_ci": [k_lo, k_hi], "t50_ci": [t_lo, t_hi], "nboot": nb_}

    print("\nADVERSARIAL fit (merged measured points, NO synthetic 200-token point):")
    popt_a, r2_a = fit_rate(tokens, av, an, [0.05, 120])
    if popt_a is not None:
        (k_lo, k_hi, k_md), (t_lo, t_hi, t_md), na_ = bootstrap_fit(tokens, av, an, [0.05, 120])
        print(f"  k={popt_a[0]:.3f}  T50={popt_a[1]:.1f}  R^2={r2_a:.3f}")
        print(f"  bootstrap k 95%CI [{k_lo:.3f},{k_hi:.3f}] median {k_md:.3f}")
        print(f"  bootstrap T50 95%CI [{t_lo:.1f},{t_hi:.1f}] median {t_md:.1f} (nboot={na_})")
        # T90 = T50 + 2.2/k
        t90 = popt_a[1] + 2.2 / popt_a[0]
        print(f"  implied T90(adv) = T50 + 2.2/k = {t90:.0f} tokens")
        result["adversarial"] = {"k": float(popt_a[0]), "t50": float(popt_a[1]), "r2": float(r2_a),
                                "k_ci": [k_lo, k_hi], "t50_ci": [t_lo, t_hi], "nboot": na_,
                                "t90": float(t90)}
        if popt_b is not None:
            ratio = popt_a[1] / popt_b[1]
            print(f"  adv/benign T50 ratio = {ratio:.2f}")
            result["t50_ratio_adv_over_benign"] = float(ratio)
    else:
        print(f"  FIT FAILED: {r2_a}")

    (RESULTS / "token_sigmoid_refit_merged.json").write_text(json.dumps(result, indent=2))
    print("\nSaved to results/token_sigmoid_refit_merged.json")


if __name__ == "__main__":
    import sys
    if "--merged" in sys.argv:
        main_merged()
    else:
        main()

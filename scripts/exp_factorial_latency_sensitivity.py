"""EXPLORATORY (v5 audit): Factorial invalid-cell analysis + latency ANOVA
sensitivity under censored/imputed latency on all 800 cells.

Zero API calls; reads factorial_results.jsonl.
"""
from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

JL = Path(__file__).resolve().parents[1] / "results" / "factorial_results.jsonl"


def load():
    rows = []
    with open(JL) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def eta2_twoway(cells):
    C = sorted({c for c, s, y in cells})
    S = sorted({s for c, s, y in cells})
    ys = np.array([y for _, _, y in cells], float)
    grand = ys.mean()
    ss_tot = np.sum((ys - grand) ** 2)
    by_c = defaultdict(list); by_s = defaultdict(list); by_cs = defaultdict(list)
    for c, s, y in cells:
        by_c[c].append(y); by_s[s].append(y); by_cs[(c, s)].append(y)
    cmean = {c: np.mean(v) for c, v in by_c.items()}
    smean = {s: np.mean(v) for s, v in by_s.items()}
    csmean = {k: np.mean(v) for k, v in by_cs.items()}
    ss_c = sum(len(by_c[c]) * (cmean[c] - grand) ** 2 for c in C)
    ss_s = sum(len(by_s[s]) * (smean[s] - grand) ** 2 for s in S)
    ss_cs = 0.0
    for c in C:
        for s in S:
            if (c, s) in csmean:
                n = len(by_cs[(c, s)])
                inter = csmean[(c, s)] - cmean[c] - smean[s] + grand
                ss_cs += n * inter ** 2
    ss_resid = ss_tot - ss_c - ss_s - ss_cs
    return {"C": ss_c / ss_tot, "S": ss_s / ss_tot,
            "CxS": ss_cs / ss_tot, "resid": ss_resid / ss_tot, "n": len(cells)}


def main():
    rows = load()
    print(f"Total rows: {len(rows)}")

    def valid(r):
        dl = r.get("detection_latency", None)
        return dl is not None and dl >= 0 and r.get("neg_clean", False)

    valid_rows = [r for r in rows if valid(r)]
    invalid_rows = [r for r in rows if not valid(r)]
    print(f"Valid: {len(valid_rows)}  Invalid: {len(invalid_rows)}")

    inv_by_shift = defaultdict(int); tot_by_shift = defaultdict(int)
    for r in rows:
        tot_by_shift[r["shift_condition"]] += 1
        if not valid(r):
            inv_by_shift[r["shift_condition"]] += 1
    print("\nInvalid by shift (MNAR check):")
    for s in sorted(tot_by_shift):
        print(f"  {s:16s}: {inv_by_shift[s]:3d}/{tot_by_shift[s]:3d} = "
              f"{inv_by_shift[s]/tot_by_shift[s]*100:.1f}%")

    inv_by_cls = defaultdict(int); tot_by_cls = defaultdict(int)
    for r in rows:
        tot_by_cls[r["classifier"]] += 1
        if not valid(r):
            inv_by_cls[r["classifier"]] += 1
    print("\nInvalid by classifier:")
    for c in sorted(tot_by_cls):
        print(f"  {c:16s}: {inv_by_cls[c]:3d}/{tot_by_cls[c]:3d} = "
              f"{inv_by_cls[c]/tot_by_cls[c]*100:.1f}%")

    base_cells = [(r["classifier"], r["shift_condition"], r["detection_latency"])
                  for r in valid_rows]
    base = eta2_twoway(base_cells)
    print(f"\nBASELINE eta^2 (valid only, n={base['n']}): "
          f"C={base['C']:.3f} S={base['S']:.3f} CxS={base['CxS']:.3f} resid={base['resid']:.3f}")
    print("  (paper: C=0.243 S=0.237 CxS=0.185 resid=0.335)")

    max_lat = max(r["detection_latency"] for r in valid_rows)
    print(f"\nMax observed latency: {max_lat}")

    for label, impute in [("max-observed", max_lat),
                          ("2x-max", 2 * max_lat), ("500", 500)]:
        cells = list(base_cells)
        for r in invalid_rows:
            cells.append((r["classifier"], r["shift_condition"], float(impute)))
        res = eta2_twoway(cells)
        print(f"SENSITIVITY impute @ {label:12s} (n={res['n']}): "
              f"C={res['C']:.3f} S={res['S']:.3f} CxS={res['CxS']:.3f} resid={res['resid']:.3f}")


if __name__ == "__main__":
    main()

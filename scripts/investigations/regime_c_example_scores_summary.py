"""Summarise results/regime_c_example_scores.json (Regime C per-example scores).

The scores were produced by regime_c_example_scores_infer.py on the Mac Studio
(DEBERTA_CHECKPOINT_PATH=checkpoints/deberta-wildguardmix) and stored, so this
summary needs no inference. The harness truncates every shifted text to 1000
characters; examples longer than that lose their adversarial suffix.
"""
import json
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[2]
d = json.load(open(R / "results/regime_c_example_scores.json"))
L = np.array(d["combined_length_chars"])
trunc = L > d["loader_truncation_chars"]
deb = np.array(d["per_example_scores_after_truncation"]["deberta"])
lg = np.array(d["per_example_scores_after_truncation"]["llama-guard"])
st = np.array(d["stored_attacked_score_untruncated"])
ref = d["reference_mean_score_500"]
print(f"examples {d['n_successful']}, longer than {d['loader_truncation_chars']} chars: {int(trunc.sum())}")
print(f"DeBERTa: stored (untruncated) attacked mean {st.mean():.4f}; as streamed {deb.mean():.4f}; "
      f"streamed scores of the truncated examples {np.round(deb[trunc], 3).tolist()}; "
      f"examples above 0.5 as streamed {int((deb > 0.5).sum())}")
print(f"Llama Guard: streamed mean {lg.mean():.4f} (reference mean {ref['llama-guard']:.4f}, "
      f"delta {lg.mean() - ref['llama-guard']:+.3f}); median {np.median(lg):.3f}; "
      f"range {lg.min():.3f} to {lg.max():.3f}; above 0.5: {int((lg > 0.5).sum())}/{len(lg)}; "
      f"distinct values {len(set(lg.round(4)))}")

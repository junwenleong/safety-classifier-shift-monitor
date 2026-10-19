import json, os, sys
cache = os.environ["SCORE_CACHE"]; out = {}
for name in sys.argv[1:]:
    d = json.load(open(os.path.join(cache, f"offsets2_{name}.json")))
    out[name] = {}
    for shift, v in d.items():
        per = {}
        for seed, hits in v["per_seed"].items():
            assert len(hits) == 1, (name, shift, seed, hits)
            per[seed] = hits[0]
        out[name][shift] = per
json.dump(out, open(os.path.join(cache, "offsets_all.json"), "w"), indent=1)
print({n: {s: sorted(set(p.values())) for s, p in v.items()} for n, v in out.items()})

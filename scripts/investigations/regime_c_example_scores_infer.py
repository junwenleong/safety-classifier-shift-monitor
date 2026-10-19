import json, os, sys
sys.path.insert(0, os.getcwd())
import scripts.run_factorial as rf
texts = [t["text"] if isinstance(t, dict) else t for t in rf.load_adversarial_success()]
out = {"n": len(texts), "texts_sha256": None}
import hashlib
out["texts_sha256"] = [hashlib.sha256(t.encode()).hexdigest() for t in texts]
for name in sys.argv[1:]:
    clf = rf.get_classifier(name)
    out[name] = [float(clf.predict(t).score) for t in texts]
    print(name, sum(out[name]) / len(out[name]), min(out[name]), max(out[name]), flush=True)
    del clf
json.dump(out, open("/tmp/regime_c_scores.json", "w"))

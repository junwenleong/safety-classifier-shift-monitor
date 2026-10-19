"""Memoising wrapper around a classifier adapter, plus helpers shared by the rerun tools.

The wrapper is applied at the classifier layer so that scripts/run_factorial.py runs unmodified.
A classifier's output for a text does not depend on anything else in the harness, so each distinct
text is scored once and stored (score, representation, metadata) keyed by SHA-256 of the text.
The cache file is a pickle per classifier, written atomically every FLUSH_EVERY misses and at close.
"""
import hashlib
import os
import pickle
import time

from shift_detection_monitor.types import ClassifierOutput

FLUSH_EVERY = 250


def key_of(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class MemoClassifier:
    def __init__(self, inner_factory, path: str):
        self._factory = inner_factory
        self._inner_obj = None
        self._path = path
        self._cache = {}
        self._dirty = 0
        self.hits = 0
        self.misses = 0
        self.infer_seconds = 0.0
        if os.path.exists(path):
            with open(path, "rb") as f:
                self._cache = pickle.load(f)

    def predict(self, text):
        k = key_of(text)
        rec = self._cache.get(k)
        if rec is not None:
            self.hits += 1
            score, rep, meta = rec
            return ClassifierOutput(score=score, representation=None if rep is None else rep.copy(), metadata=dict(meta))
        t0 = time.time()
        out = self._inner.predict(text)
        self.infer_seconds += time.time() - t0
        self.misses += 1
        rep = None if out.representation is None else out.representation.copy()
        self._cache[k] = (float(out.score), rep, dict(out.metadata))
        self._dirty += 1
        if self._dirty >= FLUSH_EVERY:
            self.flush()
        return out

    @property
    def _inner(self):
        if self._inner_obj is None:
            self._inner_obj = self._factory()
        return self._inner_obj

    def flush(self):
        tmp = self._path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(self._cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, self._path)
        self._dirty = 0

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        return getattr(self._inner, name)

    def __len__(self):
        return len(self._cache)

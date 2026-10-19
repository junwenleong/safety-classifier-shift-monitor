import random
random.seed(1)
def L(fa, fb, lam): return fa + lam * (fb - fa) ** 2
bad = 0
for _ in range(200000):
    lam = random.uniform(0.5, 8); fa = random.uniform(0.3, 0.9); fb = random.uniform(fa, 1.0); a = random.uniform(1e-3, 0.3)
    g = fb - fa
    # canary fixed, target decreases by a
    improves = L(fa - a, fb, lam) < L(fa, fb, lam)
    pred = g < 1 / (2 * lam) - a / 2
    bad += improves != pred
    # canary co-moves down by a: gap unchanged, so L must fall by exactly a
    bad += abs((L(fa, fb, lam) - L(fa - a, fb - a, lam)) - a) > 1e-12
print("mismatches:", bad)
# paper's stated assumption: delta g = -a  (gap shrinks by a, i.e. fb falls by 2a)
imp = sum(L(fa - a, fb - 2 * a, lam) < L(fa, fb, lam) for fa, fb, lam, a in [(0.6, 0.9, 2.0, 0.05), (0.6, 0.85, 2.0, 0.02), (0.6, 0.7, 4.0, 0.01)])
print("moves that lower L under delta g = -a (3 examples with g > a/2):", imp, "of 3")

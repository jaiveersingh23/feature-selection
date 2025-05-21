import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import time

def fitness(solution, X, y):
    if np.sum(solution) == 0:
        return 0
    X_selected = X[:, solution == 1]
    clf = KNeighborsClassifier(n_neighbors=5)
    return np.mean(cross_val_score(clf, X_selected, y, cv=5))

def binarize(position):
    prob = 1 / (1 + np.exp(-position))
    return (prob > 0.5).astype(int)

def OSCO(X, y, n_sperms=20, max_iter=50):
    dim = X.shape[1]
    sperms = np.random.rand(n_sperms, dim)
    ovum = np.random.rand(dim)

    binaries = np.array([binarize(s) for s in sperms])
    fitnesses = np.array([fitness(b, X, y) for b in binaries])
    best_idx = np.argmax(fitnesses)
    best_sperm = sperms[best_idx].copy()

    for _ in range(max_iter):
        for i in range(n_sperms):
            r1 = np.random.rand()
            new_pos = sperms[i] + r1 * (ovum - sperms[i]) + 0.1 * np.random.randn(dim)
            new_pos = np.clip(new_pos, 0, 1)
            new_bin = binarize(new_pos)
            new_fit = fitness(new_bin, X, y)

            if new_fit > fitnesses[i]:
                sperms[i] = new_pos
                fitnesses[i] = new_fit
                if new_fit > fitness(binarize(best_sperm), X, y):
                    best_sperm = new_pos.copy()

        # Ovum update — attracted to the current best sperm
        r2 = np.random.rand()
        ovum = ovum + r2 * (best_sperm - ovum) + 0.05 * np.random.randn(dim)
        ovum = np.clip(ovum, 0, 1)

    return binarize(best_sperm)

def run(X, y):
    start = time.time()
    best = OSCO(X, y)
    acc = fitness(best, X, y)
    return best, acc, np.sum(best), time.time() - start

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

def NOA(X, y, n_nilgais=20, max_iter=50):
    dim = X.shape[1]
    herd = np.random.rand(n_nilgais, dim)
    binaries = np.array([binarize(n) for n in herd])
    fitnesses = np.array([fitness(b, X, y) for b in binaries])
    best_idx = np.argmax(fitnesses)
    leader = herd[best_idx].copy()

    for _ in range(max_iter):
        for i in range(n_nilgais):
            jump_prob = np.random.rand()
            if jump_prob < 0.2:
                # Sudden jump (exploration)
                herd[i] = np.random.rand(dim)
            else:
                # Move towards leader + herd average + noise
                mean_herd = np.mean(herd, axis=0)
                r1, r2 = np.random.rand(), np.random.rand()
                herd[i] = herd[i] + r1 * (leader - herd[i]) + r2 * (mean_herd - herd[i]) + 0.1 * np.random.randn(dim)

            herd[i] = np.clip(herd[i], 0, 1)

        binaries = np.array([binarize(n) for n in herd])
        fitnesses = np.array([fitness(b, X, y) for b in binaries])
        current_best_idx = np.argmax(fitnesses)
        if fitnesses[current_best_idx] > fitness(binarize(leader), X, y):
            leader = herd[current_best_idx].copy()

    return binarize(leader)

def run(X, y):
    start = time.time()
    best = NOA(X, y)
    acc = fitness(best, X, y)
    return best, acc, np.sum(best), time.time() - start

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import time

def fitness(solution, X, y):
    if np.sum(solution) == 0:
        return 0
    X_selected = X[:, solution == 1]
    clf = KNeighborsClassifier(n_neighbors=5)
    return np.mean(cross_val_score(clf, X_selected, y, cv=5))

def ensure_at_least_one_feature(binary):
    if np.sum(binary) == 0:
        binary[np.random.randint(0, len(binary))] = 1
    return binary

def binarize(position):
    binary = (position > 0.5).astype(int)
    return ensure_at_least_one_feature(binary)

def KMSO(X, y, n_agents=20, max_iter=50, alpha=0.1):
    dim = X.shape[1]
    positions = np.random.rand(n_agents, dim)
    velocities = np.zeros_like(positions)

    binaries = np.array([binarize(p) for p in positions])
    fitnesses = np.array([fitness(b, X, y) for b in binaries])

    best_idx = np.argmax(fitnesses)
    best_position = positions[best_idx].copy()
    best_binary = binaries[best_idx].copy()
    best_fit = fitnesses[best_idx]

    for t in range(max_iter):
        for i in range(n_agents):
            r = np.random.rand(dim)
            acceleration = alpha * (best_position - positions[i]) * r
            velocities[i] += acceleration
            positions[i] += velocities[i] + 0.5 * acceleration * (t ** 2)
            positions[i] = np.clip(positions[i], 0, 1)

            binary = binarize(positions[i])
            fit = fitness(binary, X, y)

            if fit > fitnesses[i]:
                fitnesses[i] = fit
                binaries[i] = binary
                if fit > best_fit:
                    best_fit = fit
                    best_position = positions[i].copy()
                    best_binary = binary.copy()

    return best_binary

def run(X, y):
    start = time.time()
    best = KMSO(X, y)
    acc = fitness(best, X, y)
    return best, acc, np.sum(best), time.time() - start

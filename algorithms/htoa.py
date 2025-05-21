import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import time

def fitness(solution, X, y):
    if np.sum(solution) == 0:
        return 0
    X_selected = X[:, solution == 1]
    clf = RandomForestClassifier(n_estimators=100)
    return np.mean(cross_val_score(clf, X_selected, y, cv=5))

def ensure_at_least_one_feature(binary):
    if np.sum(binary) == 0:
        binary[np.random.randint(0, len(binary))] = 1
    return binary

def binarize(position):
    binary = (position > 0.5).astype(int)
    return ensure_at_least_one_feature(binary)

def HTOA(X, y, n_agents=20, max_iter=50, k=0.1):
    dim = X.shape[1]
    positions = np.random.rand(n_agents, dim)
    fitness_values = np.array([fitness(binarize(agent), X, y) for agent in positions])

    best_idx = np.argmax(fitness_values)
    best_position = positions[best_idx].copy()
    best_fit = fitness_values[best_idx]

    for t in range(max_iter):
        for i in range(n_agents):
            temp_diff = best_fit - fitness_values[i]
            heat_transfer = k * (temp_diff / (np.linalg.norm(best_position - positions[i]) + 1e-6))  # To avoid division by zero

            positions[i] += heat_transfer * (best_position - positions[i])
            positions[i] = np.clip(positions[i], 0, 1)

            binary = binarize(positions[i])
            fit = fitness(binary, X, y)

            fitness_values[i] = fit

            if fit > best_fit:
                best_fit = fit
                best_position = positions[i].copy()

    return best_position

def run(X, y):
    start = time.time()
    best = HTOA(X, y)
    acc = fitness(best, X, y)
    return best, acc, np.sum(best), time.time() - start

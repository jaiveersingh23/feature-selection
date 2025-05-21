import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import time

def fitness(solution, X, y):
    """Evaluate the fitness (accuracy) of a binary feature subset."""
    if np.sum(solution) == 0:
        return 0
    X_selected = X[:, solution == 1]
    clf = RandomForestClassifier(n_estimators=50, n_jobs=-1)
    return np.mean(cross_val_score(clf, X_selected, y, cv=3, n_jobs=-1))

def ensure_at_least_one_feature(solution):
    """Ensure that at least one feature is selected."""
    if np.sum(solution) == 0:
        idx = np.random.randint(0, len(solution))
        solution[idx] = 1
    return solution

def binarize(position):
    """Convert continuous position to binary feature selector using sigmoid."""
    sigmoid = 1 / (1 + np.exp(-position))
    binary = (sigmoid > 0.5).astype(int)
    return ensure_at_least_one_feature(binary)

def ACO_update(ant, best):
    """Generate a new position vector using ACO-inspired update rule."""
    r1, r2 = np.random.rand(), np.random.rand()
    new_position = ant + r1 * (best - ant)
    return np.clip(new_position, 0, 1)

def ACO(X, y, n_agents=10, max_iter=20):
    dim = X.shape[1]
    population = np.random.rand(n_agents, dim)
    binaries = np.array([binarize(ind) for ind in population])
    fitnesses = np.array([fitness(b, X, y) for b in binaries])
    best_idx = np.argmax(fitnesses)
    best_binary = binaries[best_idx].copy()
    best_position = population[best_idx].copy()

    for _ in range(max_iter):
        for i in range(n_agents):
            new_position = ACO_update(population[i], best_position)
            new_binary = binarize(new_position)
            new_fit = fitness(new_binary, X, y)

            if new_fit > fitnesses[i]:
                population[i] = new_position
                binaries[i] = new_binary
                fitnesses[i] = new_fit

                if new_fit > fitness(best_binary, X, y):
                    best_binary = new_binary.copy()
                    best_position = new_position.copy()

    return best_binary

def run(X, y):
    start = time.time()
    best = ACO(X, y)
    acc = fitness(best, X, y)
    return best, acc, np.sum(best), time.time() - start

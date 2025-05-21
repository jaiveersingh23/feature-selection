import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import time

def fitness(solution, X, y):
    """Evaluate fitness based on selected features."""
    if np.sum(solution) == 0:
        return 0  # Avoid empty feature sets
    X_selected = X[:, solution == 1]
    clf = RandomForestClassifier(n_estimators=100)
    return np.mean(cross_val_score(clf, X_selected, y, cv=5))


def binarize(position):
    """Convert continuous to binary using 0.5 threshold."""
    return (position > 0.5).astype(int)

def bengal_tiger_hunting_optimization(X, y, n_individuals=20, max_iter=50, elite_frac=0.2, pounce_factor=0.2, mutation_rate=0.05):
    dim = X.shape[1]
    population = np.random.rand(n_individuals, dim)
    binaries = np.array([binarize(ind) for ind in population])
    fitnesses = np.array([fitness(ind, X, y) for ind in binaries])

    best_idx = np.argmax(fitnesses)
    best_solution = binaries[best_idx].copy()
    best_score = fitnesses[best_idx]

    for _ in range(max_iter):
        sorted_idx = np.argsort(fitnesses)[::-1]
        population = population[sorted_idx]
        binaries = binaries[sorted_idx]
        fitnesses = fitnesses[sorted_idx]

        n_elite = int(elite_frac * n_individuals)
        elite = population[:n_elite]

        # Stalking (Exploration)
        for i in range(n_elite, n_individuals):
            new_position = population[i] + pounce_factor * (np.random.rand(dim) - 0.5)
            new_position = np.clip(new_position, 0, 1)

            if np.random.rand() < mutation_rate:
                mutation = np.random.normal(0, 0.2, dim)
                new_position += mutation
                new_position = np.clip(new_position, 0, 1)

            new_binary = binarize(new_position)
            fit = fitness(new_binary, X, y)
            population[i] = new_position
            binaries[i] = new_binary
            fitnesses[i] = fit

            # Pouncing (Exploitation)
            if fit > best_score:
                best_solution = new_binary.copy()
                best_score = fit

        # Targeting (Convergence)
        if best_score > np.mean(fitnesses):
            break

    return best_solution

def run(X, y):
    start = time.time()
    best = bengal_tiger_hunting_optimization(X, y)
    acc = fitness(best, X, y)
    return best, acc, np.sum(best), time.time() - start

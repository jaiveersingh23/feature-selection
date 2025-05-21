import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import time

def fitness(solution, X, y):
    """Evaluate fitness using classification accuracy."""
    if np.sum(solution) == 0:
        return 0
    X_selected = X[:, solution == 1]
    clf = KNeighborsClassifier(n_neighbors=5)
    return np.mean(cross_val_score(clf, X_selected, y, cv=5))

def ensure_at_least_one_feature(solution):
    """Ensure that at least one feature is selected."""
    if np.sum(solution) == 0:
        solution[np.random.randint(0, len(solution))] = 1
    return solution
"""
def binarize(position):
    #Binarize a continuous vector using sigmoid.
    sigmoid = 1 / (1 + np.exp(-position))
    binary = (sigmoid > 0.5).astype(int)
    return ensure_at_least_one_feature(binary)
"""
def binarize(position):
    """Convert continuous to binary using 0.5 threshold."""
    return (position > 0.5).astype(int)

def POA(X, y, n_peacocks=20, max_iter=50, elite_frac=0.2, mutation_rate=0.1):
    """Peacock Optimization Algorithm for feature selection."""
    dim = X.shape[1]
    population = np.random.rand(n_peacocks, dim)
    binaries = np.array([binarize(ind) for ind in population])
    fitnesses = np.array([fitness(b, X, y) for b in binaries])

    best_idx = np.argmax(fitnesses)
    best_solution = binaries[best_idx].copy()
    best_score = fitnesses[best_idx]

    for _ in range(max_iter):
        # Sort by fitness
        sorted_idx = np.argsort(fitnesses)[::-1]
        population = population[sorted_idx]
        binaries = binaries[sorted_idx]
        fitnesses = fitnesses[sorted_idx]

        n_elite = int(elite_frac * n_peacocks)
        elite = population[:n_elite]

        # Update non-elite peacocks
        for i in range(n_elite, n_peacocks):
            leader = elite[np.random.randint(0, n_elite)]
            beta = np.random.normal(0, 0.1, dim)
            move = population[i] + beta * (leader - population[i])

            # Mutation
            if np.random.rand() < mutation_rate:
                mutation = np.random.normal(0, 0.2, dim)
                move += mutation

            move = np.clip(move, 0, 1)
            binary_move = binarize(move)
            fit = fitness(binary_move, X, y)

            population[i] = move
            binaries[i] = binary_move
            fitnesses[i] = fit

            if fit > best_score:
                best_solution = binary_move.copy()
                best_score = fit

    return best_solution

def run(X, y):
    """Run POA and return best feature subset, accuracy, feature count, and time."""
    start = time.time()
    best = POA(X, y)
    acc = fitness(best, X, y)
    return best, acc, np.sum(best), time.time() - start

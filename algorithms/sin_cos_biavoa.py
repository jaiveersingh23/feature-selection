import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import time

np.random.seed(42)  # For reproducibility

# Sigmoid-based binarization
def sigmoid_binarize(x):
    return (1 / (1 + np.exp(-x))) > 0.5

# Fitness function with sparsity penalty and feature-check
def fitness(solution, X, y, alpha=0.0):
    if np.sum(solution) == 0:  # No features selected — invalid
        return 0
    X_selected = X[:, solution == 1]
    try:
        clf = KNeighborsClassifier(n_neighbors=5)
        scores = cross_val_score(clf, X_selected, y, cv=5)
        acc = np.mean(scores)
    except ValueError:
        return 0  # Avoid model crash on empty feature sets
    penalty = alpha * (np.sum(solution) / len(solution))
    return acc - penalty

# Position update function
def update_position(current, best):
    r1, r2 = np.random.rand(), np.random.rand()
    A = 2 * r1 - 1
    C = 2 * r2
    new_position = current + A * (best - current)
    return np.clip(new_position, 0, 1)

# Main bIAVOA optimization
def bIAVOA(X, y, n_agents=20, max_iter=50, alpha=0.0):
    dim = X.shape[1]
    population = np.random.rand(n_agents, dim)

    # Initial evaluation
    bin_pop = np.array([sigmoid_binarize(ind).astype(int) for ind in population])
    # Ensure at least 1 feature is selected in each individual
    for i in range(n_agents):
        if np.sum(bin_pop[i]) == 0:
            bin_pop[i][np.random.randint(dim)] = 1

    fitnesses = np.array([fitness(ind, X, y, alpha) for ind in bin_pop])
    best_idx = np.argmax(fitnesses)
    best = bin_pop[best_idx].copy()
    best_fitness = fitnesses[best_idx]

    # Main loop
    for _ in range(max_iter):
        for i in range(n_agents):
            new_position = update_position(population[i], best)
            bin_new_position = sigmoid_binarize(new_position).astype(int)

            if np.sum(bin_new_position) == 0:
                bin_new_position[np.random.randint(dim)] = 1

            new_fit = fitness(bin_new_position, X, y, alpha)

            if new_fit > fitnesses[i]:
                population[i] = new_position
                fitnesses[i] = new_fit

                if new_fit > best_fitness:
                    best = bin_new_position.copy()
                    best_fitness = new_fit

    return best, best_fitness

# Wrapper with timing and result structure
def run(X, y, alpha=0.0):
    start = time.time()
    best_solution, best_acc = bIAVOA(X, y, alpha=alpha)
    duration = time.time() - start
    return best_solution, float(best_acc), int(np.sum(best_solution)), float(time.time() - start)


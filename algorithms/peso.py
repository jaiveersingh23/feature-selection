import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import time

def fitness(solution, X, y):
    """Evaluate fitness as classification accuracy using selected features."""
    if np.sum(solution) == 0:
        return 0
    X_selected = X[:, solution == 1]
    clf = KNeighborsClassifier(n_neighbors=5)
    return np.mean(cross_val_score(clf, X_selected, y, cv=5))

def ensure_at_least_one_feature(solution):
    """Ensure that at least one feature is selected."""
    if np.sum(solution) == 0:
        idx = np.random.randint(0, len(solution))
        solution[idx] = 1
    return solution

def binarize(position):
    """Convert continuous position to binary using sigmoid and threshold."""
    sigmoid = 1 / (1 + np.exp(-position))
    binary = (sigmoid > 0.5).astype(int)
    return ensure_at_least_one_feature(binary)

def PESO(X, y, n_agents=20, max_iter=50, α=0.3, β=0.4, γ=0.2):
    """Polyp Expansion and Symbiosis Optimization algorithm."""
    dim = X.shape[1]
    population = np.random.uniform(0, 1, (n_agents, dim))
    binaries = np.array([binarize(pos) for pos in population])
    fitnesses = np.array([fitness(b, X, y) for b in binaries])
    
    best_idx = np.argmax(fitnesses)
    best_position = population[best_idx].copy()
    best_binary = binaries[best_idx].copy()
    best_fitness = fitnesses[best_idx]

    for _ in range(max_iter):
        for i in range(n_agents):
            # Expansion
            expanded = population[i] + α * np.random.randn(dim)
            expanded = np.clip(expanded, 0, 1)
            
            # Symbiosis (move toward best)
            symbiotic = expanded + β * (best_position - population[i])
            symbiotic = np.clip(symbiotic, 0, 1)

            # Evaluate new position
            new_position = symbiotic.copy()
            new_binary = binarize(new_position)
            new_fitness = fitness(new_binary, X, y)

            # Contraction if no improvement
            if new_fitness <= fitnesses[i]:
                new_position = population[i] - γ * np.abs(population[i] - np.random.rand(dim))
                new_position = np.clip(new_position, 0, 1)
                new_binary = binarize(new_position)
                new_fitness = fitness(new_binary, X, y)

            # Accept new position if better
            if new_fitness > fitnesses[i]:
                population[i] = new_position
                binaries[i] = new_binary
                fitnesses[i] = new_fitness

                if new_fitness > best_fitness:
                    best_position = new_position.copy()
                    best_binary = new_binary.copy()
                    best_fitness = new_fitness

        # Reef Drift: replace worst agent with random new one
        worst_idx = np.argmin(fitnesses)
        new_pos = np.random.rand(dim)
        new_bin = binarize(new_pos)
        new_fit = fitness(new_bin, X, y)

        population[worst_idx] = new_pos
        binaries[worst_idx] = new_bin
        fitnesses[worst_idx] = new_fit

    return best_binary

def run(X, y):
    """Run PESO and return best feature subset, accuracy, count, and time."""
    start = time.time()
    best = PESO(X, y)
    acc = fitness(best, X, y)
    return best, acc, np.sum(best), time.time() - start

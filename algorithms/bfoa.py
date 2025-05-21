import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import time

def fitness(solution, X, y):
    """Evaluate the fitness (accuracy) of a binary feature subset using KNN."""
    if np.sum(solution) == 0:
        return 0
    X_selected = X[:, solution == 1]
    clf = KNeighborsClassifier(n_neighbors=5)
    return np.mean(cross_val_score(clf, X_selected, y, cv=5))

def binarize(position):
    """Convert continuous to binary using 0.5 threshold."""
    binary = (position > 0.5).astype(int)
    if np.sum(binary) == 0:
        binary[np.random.randint(0, len(binary))] = 1  # Ensure at least one feature
    return binary

def bee_foraging_optimization(X, y, n_bees=20, max_iter=50, elite_frac=0.2, mutation_rate=0.1, waggle_factor=0.1):
    """Main loop for Bee Foraging Optimization Algorithm."""
    dim = X.shape[1]
    population = np.random.rand(n_bees, dim)
    fitness_values = np.array([fitness(binarize(bee), X, y) for bee in population])

    best_idx = np.argmax(fitness_values)
    best_bee = population[best_idx].copy()
    best_fitness = fitness_values[best_idx]

    for _ in range(max_iter):
        sorted_indices = np.argsort(fitness_values)[::-1]
        population = population[sorted_indices]
        fitness_values = fitness_values[sorted_indices]

        n_elite = max(1, int(elite_frac * n_bees))
        elite = population[:n_elite]

        for i in range(n_elite, n_bees):
            leader = elite[np.random.randint(0, n_elite)]
            move = leader + waggle_factor * (np.random.rand(dim) - 0.5)
            move = np.clip(move, 0, 1)

            if np.random.rand() < mutation_rate:
                move += np.random.normal(0, 0.2, dim)
                move = np.clip(move, 0, 1)

            bin_move = binarize(move)
            move_fitness = fitness(bin_move, X, y)

            population[i] = move
            fitness_values[i] = move_fitness

            if move_fitness > best_fitness:
                best_fitness = move_fitness
                best_bee = move.copy()

    return best_bee

def run(X, y):
    """Run BFOA and return best solution, accuracy, feature count, and time taken."""
    start = time.time()
    best = bee_foraging_optimization(X, y)
    acc = fitness(binarize(best), X, y)
    return best, acc, np.sum(binarize(best)), time.time() - start

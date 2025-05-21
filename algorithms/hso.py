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

def binarize(position):
    prob = 1 / (1 + np.exp(-position))
    return (prob > 0.5).astype(int)

def initialize_swarm(X, y, n_bees=50):
    dim = X.shape[1]
    swarm = np.random.rand(n_bees, dim)
    binaries = np.array([binarize(bee) for bee in swarm])
    fitness_scores = [fitness(b, X, y) for b in binaries]
    best_idx = np.argmax(fitness_scores)
    return swarm, fitness_scores, swarm[best_idx].copy()

def update_bee_position(bee, queen_bee, exploration_factor=0.5):
    r1, r2 = np.random.rand(), np.random.rand()
    new_position = bee + r1 * (queen_bee - bee) + r2 * exploration_factor * np.random.randn(bee.shape[0])
    return np.clip(new_position, 0, 1)

def HSO(X, y, max_iter=50, n_bees=20):
    swarm, fitness_scores, best_bee = initialize_swarm(X, y, n_bees)
    best_fitness = max(fitness_scores)

    for _ in range(max_iter):
        for i in range(n_bees):
            new_position = update_bee_position(swarm[i], best_bee)
            new_binary = binarize(new_position)
            new_fit = fitness(new_binary, X, y)

            if new_fit > fitness_scores[i]:
                swarm[i] = new_position
                fitness_scores[i] = new_fit

                if new_fit > best_fitness:
                    best_bee = new_position
                    best_fitness = new_fit

    return binarize(best_bee)

def run(X, y):
    start = time.time()
    best = HSO(X, y)
    acc = fitness(best, X, y)
    return best, acc, np.sum(best), time.time() - start

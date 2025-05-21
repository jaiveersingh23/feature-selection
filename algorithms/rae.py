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

def erosion(solution, erosion_rate=0.1):
    if np.sum(solution) <= 1:
        return solution
    indices = np.where(solution == 1)[0]
    erosion_count = max(1, int(len(indices) * erosion_rate))
    to_remove = np.random.choice(indices, erosion_count, replace=False)
    solution[to_remove] = 0
    return solution

def deposition(position, best_position, rate=0.2):
    return position + rate * (best_position - position)

def RAE(X, y, n_droplets=20, max_iter=50):
    dim = X.shape[1]
    population = np.random.rand(n_droplets, dim)
    binaries = np.array([binarize(ind) for ind in population])
    fitnesses = np.array([fitness(b, X, y) for b in binaries])
    best_idx = np.argmax(fitnesses)
    best_binary = binaries[best_idx].copy()
    best_position = population[best_idx].copy()

    for _ in range(max_iter):
        for i in range(n_droplets):
            binary = binarize(population[i])
            eroded = erosion(binary.copy())
            new_position = population[i] * eroded
            new_position = deposition(new_position, best_position)
            new_position = np.clip(new_position, 0, 1)
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
    best = RAE(X, y)
    acc = fitness(best, X, y)
    return best, acc, np.sum(best), time.time() - start

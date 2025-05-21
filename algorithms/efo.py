import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
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

def coulomb_force(q1, q2, r):
    k = 8.99e9
    return k * (q1 * q2) / (r ** 2 + 1e-6)

def EFO(X, y, n_particles=20, max_iter=50):
    dim = X.shape[1]
    population = np.random.rand(n_particles, dim)
    binaries = np.array([binarize(p) for p in population])
    fitnesses = np.array([fitness(b, X, y) for b in binaries])
    best_idx = np.argmax(fitnesses)
    best_binary = binaries[best_idx].copy()
    best_position = population[best_idx].copy()

    for _ in range(max_iter):
        new_population = []
        for i in range(n_particles):
            force = np.zeros(dim)
            for j in range(n_particles):
                if i != j:
                    r = np.linalg.norm(population[i] - population[j])
                    q1, q2 = fitnesses[i], fitnesses[j]
                    direction = population[j] - population[i]
                    f = coulomb_force(q1, q2, r)
                    force += f * direction
            new_position = population[i] + 0.001 * force
            new_position = np.clip(new_position, 0, 1)
            new_population.append(new_position)

        population = np.array(new_population)
        binaries = np.array([binarize(p) for p in population])
        fitnesses = np.array([fitness(b, X, y) for b in binaries])

        current_best_idx = np.argmax(fitnesses)
        if fitnesses[current_best_idx] > fitness(best_binary, X, y):
            best_binary = binaries[current_best_idx].copy()
            best_position = population[current_best_idx].copy()

    return best_binary

def run(X, y):
    start = time.time()
    best = EFO(X, y)
    acc = fitness(best, X, y)
    return best, acc, np.sum(best), time.time() - start

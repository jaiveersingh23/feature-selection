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
    sigmoid = 1 / (1 + np.exp(-position))
    binary = (sigmoid > 0.5).astype(int)
    if np.sum(binary) == 0:
        binary[np.random.randint(len(binary))] = 1
    return binary

def STO(X, y, n_agents=20, max_iter=50, gamma=0.5, alpha_range=(0.5, 1.0), beta=0.1):
    dim = X.shape[1]
    positions = np.random.rand(n_agents, dim)
    binaries = np.array([binarize(p) for p in positions])
    fitnesses = np.array([fitness(b, X, y) for b in binaries])

    best_idx = np.argmax(fitnesses)
    best_solution = binaries[best_idx].copy()
    best_fitness = fitnesses[best_idx]

    for _ in range(max_iter):
        for i in range(n_agents):
            j = np.random.randint(0, n_agents)
            if i == j:
                continue

            fi, fj = fitnesses[i], fitnesses[j]
            tunneling_prob = np.exp(-abs(fi - fj) / gamma)

            if np.random.rand() < tunneling_prob:
                alpha = np.random.uniform(*alpha_range)
                noise = np.random.randn(dim) * beta
                new_position = positions[i] + alpha * (positions[j] - positions[i]) + noise
                new_position = np.clip(new_position, 0, 1)

                new_binary = binarize(new_position)
                new_fit = fitness(new_binary, X, y)

                if new_fit > fitnesses[i]:
                    positions[i] = new_position
                    binaries[i] = new_binary
                    fitnesses[i] = new_fit

                    if new_fit > best_fitness:
                        best_solution = new_binary.copy()
                        best_fitness = new_fit

    return best_solution

def run(X, y):
    start = time.time()
    best = STO(X, y)
    acc = fitness(best, X, y)
    return best, acc, np.sum(best), time.time() - start

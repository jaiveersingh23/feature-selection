import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import time

def fitness(solution, X, y):
    if np.sum(solution) == 0:
        return 0  # Defensive fallback
    X_selected = X[:, solution == 1]
    clf = KNeighborsClassifier(n_neighbors=5)
    return np.mean(cross_val_score(clf, X_selected, y, cv=5))

def binarize(position):
    sigmoid = 1 / (1 + np.exp(-position))
    binary = (sigmoid > 0.5).astype(int)
    return binary

def ripple_effect(solution, best_solution, r):
    distance = best_solution - solution
    ripple = r * distance
    new_solution = solution + ripple

    # Add random noise for global exploration
    noise = np.random.rand(solution.shape[0]) * 0.1
    new_solution += noise

    return np.clip(new_solution, 0, 1)

def WROA(X, y, n_agents=20, max_iter=50):
    dim = X.shape[1]
    population = np.random.rand(n_agents, dim)
    binaries = np.array([binarize(ind) for ind in population])
    fitnesses = np.array([fitness(b, X, y) for b in binaries])
    
    best_idx = np.argmax(fitnesses)
    best_cont = population[best_idx].copy()

    for _ in range(max_iter):
        for i in range(n_agents):
            r = np.random.rand() * 0.1  # Ripple intensity
            new_cont = ripple_effect(population[i], best_cont, r)
            new_bin = binarize(new_cont)
            new_fit = fitness(new_bin, X, y)

            if new_fit > fitnesses[i]:
                population[i] = new_cont
                binaries[i] = new_bin
                fitnesses[i] = new_fit

                if new_fit > fitness(binarize(best_cont), X, y):
                    best_cont = new_cont.copy()

    return binarize(best_cont)

def run(X, y):
    start = time.time()
    best = WROA(X, y)
    acc = fitness(best, X, y)
    return best, acc, np.sum(best), time.time() - start

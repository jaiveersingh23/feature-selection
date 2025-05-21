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

def sunflower_pollination(local_solution, global_solution):
    r_local = np.random.rand()
    new_local_solution = local_solution + r_local * (global_solution - local_solution)
    new_local_solution = np.clip(new_local_solution, 0, 1)

    if np.random.rand() < 0.1:  # 10% chance to explore globally
        r_global = np.random.rand()
        random_solution = np.random.rand(local_solution.shape[0])
        new_local_solution = local_solution + r_global * (random_solution - local_solution)

    new_local_solution = np.clip(new_local_solution, 0, 1)
    return new_local_solution

def SPA(X, y, n_agents=20, max_iter=50):
    dim = X.shape[1]
    population = np.random.rand(n_agents, dim)
    binaries = np.array([binarize(ind) for ind in population])
    fitnesses = np.array([fitness(b, X, y) for b in binaries])
    
    best_idx = np.argmax(fitnesses)
    global_solution = population[best_idx].copy()

    for _ in range(max_iter):
        for i in range(n_agents):
            new_cont = sunflower_pollination(population[i], global_solution)
            new_bin = binarize(new_cont)
            new_fit = fitness(new_bin, X, y)

            if new_fit > fitnesses[i]:
                population[i] = new_cont
                binaries[i] = new_bin
                fitnesses[i] = new_fit

                if new_fit > fitness(binarize(global_solution), X, y):
                    global_solution = new_cont.copy()

    return binarize(global_solution)

def run(X, y):
    start = time.time()
    best = SPA(X, y)
    acc = fitness(best, X, y)
    return best, acc, np.sum(best), time.time() - start

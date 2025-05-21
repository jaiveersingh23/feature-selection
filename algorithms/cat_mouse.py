import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import time

def fitness(solution, X, y):
    if np.sum(solution) == 0:
        return 0
    X_selected = X[:, solution == 1]
    clf = KNeighborsClassifier(n_neighbors=5)
    score = np.mean(cross_val_score(clf, X_selected, y, cv=5))
    return score

def CatAndMouse(X, y, n_agents=20, max_iter=50):
    dim = X.shape[1]
    population = np.random.randint(0, 2, (n_agents, dim))
    fitnesses = np.array([fitness(ind, X, y) for ind in population])
    best = population[np.argmax(fitnesses)].copy()

    for _ in range(max_iter):
        sorted_idx = np.argsort(-fitnesses)
        n_half = n_agents // 2
        cats = population[sorted_idx[:n_half]]
        mice = population[sorted_idx[n_half:]]

        # Cat update
        for i in range(n_half):
            r = np.random.rand(dim)
            new_cat = cats[i] + r * (best - cats[i])
            new_cat = (1 / (1 + np.exp(-new_cat))) > 0.5
            new_cat = new_cat.astype(int)
            new_fit = fitness(new_cat, X, y)
            if new_fit > fitnesses[sorted_idx[i]]:
                population[sorted_idx[i]] = new_cat
                fitnesses[sorted_idx[i]] = new_fit
                if new_fit > fitness(best, X, y):
                    best = new_cat.copy()

        # Mouse update
        for i in range(n_half, n_agents):
            r = np.random.rand(dim)
            new_mouse = population[sorted_idx[i]] + r * (population[np.random.randint(n_agents)] - population[sorted_idx[i]])
            new_mouse = (1 / (1 + np.exp(-new_mouse))) > 0.5
            new_mouse = new_mouse.astype(int)
            new_fit = fitness(new_mouse, X, y)
            if new_fit > fitnesses[sorted_idx[i]]:
                population[sorted_idx[i]] = new_mouse
                fitnesses[sorted_idx[i]] = new_fit
                if new_fit > fitness(best, X, y):
                    best = new_mouse.copy()

    return best

def run(X, y):
    start = time.time()
    best = CatAndMouse(X, y)
    acc = fitness(best, X, y)
    return best, acc, np.sum(best), time.time() - start

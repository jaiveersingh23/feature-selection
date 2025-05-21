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

def SSO(X, y, n_agents=20, max_iter=50):
    dim = X.shape[1]
    population = np.random.randint(0, 2, (n_agents, dim))
    fitnesses = np.array([fitness(ind, X, y) for ind in population])
    best_idx = np.argmax(fitnesses)
    best = population[best_idx].copy()

    for _ in range(max_iter):
        for i in range(n_agents):
            sperm = population[i]
            mutation = np.random.rand(dim) < 0.1
            new_sperm = sperm ^ mutation.astype(int)
            new_fit = fitness(new_sperm, X, y)
            if new_fit > fitnesses[i]:
                population[i] = new_sperm
                fitnesses[i] = new_fit
                if new_fit > fitness(best, X, y):
                    best = new_sperm.copy()
    return best

def run(X, y):
    start = time.time()
    best = SSO(X, y)
    acc = fitness(best, X, y)
    return best, acc, np.sum(best), time.time() - start

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import time

def fitness(solution, X, y):
    if np.sum(solution) == 0:
        return 0
    X_selected = X[:, solution == 1]
    clf = KNeighborsClassifier(n_neighbors=5)
    score = np.mean(cross_val_score(clf, X_selected, y, cv=5))
    return score

def HOA(X, y, n_agents=20, max_iter=50):
    dim = X.shape[1]
    horses = np.random.randint(0, 2, (n_agents, dim))
    fitnesses = np.array([fitness(ind, X, y) for ind in horses])
    best = horses[np.argmax(fitnesses)].copy()

    for _ in range(max_iter):
        for i in range(n_agents):
            rand_horse = horses[np.random.randint(n_agents)]
            alpha = np.random.uniform(0.1, 0.5)
            new_horse = (horses[i] + alpha * (best - rand_horse)).astype(int)
            new_horse = np.clip(new_horse, 0, 1)
            new_fit = fitness(new_horse, X, y)
            if new_fit > fitnesses[i]:
                horses[i] = new_horse
                fitnesses[i] = new_fit
                if new_fit > fitness(best, X, y):
                    best = new_horse.copy()
    return best

def run(X, y):
    start = time.time()
    best = HOA(X, y)
    acc = fitness(best, X, y)
    return best, acc, np.sum(best), time.time() - start

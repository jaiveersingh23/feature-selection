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

def GGO(X, y, n_agents=20, max_iter=50):
    dim = X.shape[1]
    geese = np.random.randint(0, 2, (n_agents, dim))
    fitnesses = np.array([fitness(ind, X, y) for ind in geese])
    best = geese[np.argmax(fitnesses)].copy()

    for _ in range(max_iter):
        for i in range(n_agents):
            leader = geese[np.random.randint(n_agents)]
            step = np.random.uniform(-1, 1, dim)
            new_goose = geese[i] + step * (leader - geese[i])
            new_goose = (1 / (1 + np.exp(-new_goose))) > 0.5
            new_goose = new_goose.astype(int)
            new_fit = fitness(new_goose, X, y)
            if new_fit > fitnesses[i]:
                geese[i] = new_goose
                fitnesses[i] = new_fit
                if new_fit > fitness(best, X, y):
                    best = new_goose.copy()
    return best

def run(X, y):
    start = time.time()
    best = GGO(X, y)
    acc = fitness(best, X, y)
    return best, acc, np.sum(best), time.time() - start

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

def HBA(X, y, n_agents=20, max_iter=50):
    dim = X.shape[1]
    pop = np.random.randint(0, 2, (n_agents, dim))
    fit = np.array([fitness(ind, X, y) for ind in pop])
    best = pop[np.argmax(fit)].copy()

    for t in range(1, max_iter + 1):
        I = 2 * np.random.rand() * (1 - (t / max_iter))  # intensification
        for i in range(n_agents):
            F = np.random.rand(dim)
            r1 = np.random.rand()
            A = I * (2 * r1 - 1)
            if np.random.rand() < 0.5:
                new_sol = best + A * np.abs(F * best - pop[i])
            else:
                new_sol = pop[i] + A * np.abs(F * best - pop[i])
            new_sol = (1 / (1 + np.exp(-new_sol))) > 0.5
            new_sol = new_sol.astype(int)

            new_fit = fitness(new_sol, X, y)
            if new_fit > fit[i]:
                pop[i] = new_sol
                fit[i] = new_fit
                if new_fit > fitness(best, X, y):
                    best = new_sol.copy()
    return best

def run(X, y):
    start = time.time()
    best = HBA(X, y)
    acc = fitness(best, X, y)
    return best, acc, np.sum(best), time.time() - start

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

def collapse(prob_vector):
    binary = (np.random.rand(*prob_vector.shape) < prob_vector).astype(int)
    if np.sum(binary) == 0:
        binary[np.random.randint(0, len(binary))] = 1
    return binary

def QSSO(X, y, n_agents=20, max_iter=50, lambda_range=(0.5, 0.9), sigma=0.05):
    dim = X.shape[1]
    agents = np.random.rand(n_agents, dim)
    binaries = np.array([collapse(agent) for agent in agents])
    fitnesses = np.array([fitness(b, X, y) for b in binaries])

    best_idx = np.argmax(fitnesses)
    best_vector = agents[best_idx].copy()
    best_binary = binaries[best_idx].copy()
    best_fit = fitnesses[best_idx]

    for _ in range(max_iter):
        for i in range(n_agents):
            lam = np.random.uniform(*lambda_range)
            noise = np.random.normal(0, sigma, dim)
            agents[i] = lam * agents[i] + (1 - lam) * best_vector + noise
            agents[i] = np.clip(agents[i], 0, 1)

            binary = collapse(agents[i])
            fit = fitness(binary, X, y)

            if fit > fitnesses[i]:
                fitnesses[i] = fit
                binaries[i] = binary

                if fit > best_fit:
                    best_fit = fit
                    best_vector = agents[i].copy()
                    best_binary = binary.copy()

    return best_binary

def run(X, y):
    start = time.time()
    best = QSSO(X, y)
    acc = fitness(best, X, y)
    return best, acc, np.sum(best), time.time() - start

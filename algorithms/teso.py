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
    binary = (1 / (1 + np.exp(-position))) > 0.5
    binary = binary.astype(int)
    if np.sum(binary) == 0:
        binary[np.random.randint(0, len(binary))] = 1
    return binary

def TESO(X, y, n_agents=20, max_iter=50, T0=1.0, alpha=0.95, delta=1.0, beta=0.3):
    dim = X.shape[1]
    agents = np.random.uniform(0, 1, (n_agents, dim))
    binaries = np.array([binarize(agent) for agent in agents])
    fitnesses = np.array([fitness(b, X, y) for b in binaries])

    best_idx = np.argmax(fitnesses)
    best_agent = agents[best_idx].copy()
    best_binary = binaries[best_idx].copy()
    best_fit = fitnesses[best_idx]

    for t in range(max_iter):
        T = T0 * (alpha ** t)
        for i in range(n_agents):
            r = np.random.rand(dim)
            expansion = delta * T * (r - 0.5)
            agents[i] = agents[i] + expansion
            agents[i] = agents[i] + beta * (best_agent - agents[i])
            agents[i] = np.clip(agents[i], 0, 1)

            binary = binarize(agents[i])
            fit = fitness(binary, X, y)

            if fit > fitnesses[i]:
                fitnesses[i] = fit
                binaries[i] = binary
                if fit > best_fit:
                    best_fit = fit
                    best_agent = agents[i].copy()
                    best_binary = binary.copy()

    return best_binary

def run(X, y):
    start = time.time()
    best = TESO(X, y)
    acc = fitness(best, X, y)
    return best, acc, np.sum(best), time.time() - start

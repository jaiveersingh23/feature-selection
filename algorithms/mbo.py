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

def bargaining_offer(agent, best, alpha=0.1):
    """Update agent's solution toward the best with step size alpha (bitwise)"""
    flip = np.random.rand(len(agent)) < alpha
    new_agent = agent.copy()
    new_agent[flip] = best[flip]
    return new_agent

def negotiation(agents, fitnesses, X, y, threshold=0.01, alpha=0.1):
    n_agents = len(agents)
    new_agents = agents.copy()
    accepted = np.zeros(n_agents, dtype=bool)

    for i in range(n_agents):
        best_offer = agents[i]
        max_improvement = 0
        for j in range(n_agents):
            if i != j:
                fit_j = fitnesses[j]
                improvement = fit_j - fitnesses[i]
                if improvement > threshold and improvement > max_improvement:
                    best_offer = agents[j]
                    max_improvement = improvement
        if not np.array_equal(best_offer, agents[i]):
            new_agents[i] = bargaining_offer(agents[i], best_offer, alpha)
            accepted[i] = True
    return new_agents, accepted

def MBO(X, y, n_agents=20, max_iter=50, alpha=0.1):
    dim = X.shape[1]
    agents = np.random.randint(0, 2, (n_agents, dim))
    fitnesses = np.array([fitness(agent, X, y) for agent in agents])
    best_idx = np.argmax(fitnesses)
    best = agents[best_idx].copy()

    for _ in range(max_iter):
        agents, accepted = negotiation(agents, fitnesses, X, y, alpha=alpha)
        fitnesses = np.array([fitness(agent, X, y) for agent in agents])
        current_best_idx = np.argmax(fitnesses)
        if fitnesses[current_best_idx] > fitness(best, X, y):
            best = agents[current_best_idx].copy()
        if np.all(accepted):  # Optional convergence condition
            break
    return best

def run(X, y):
    start = time.time()
    best = MBO(X, y)
    acc = fitness(best, X, y)
    return best, acc, np.sum(best), time.time() - start

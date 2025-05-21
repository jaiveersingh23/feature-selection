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

def CDMO(X, y, n_agents=20, max_iter=50, alpha=0.1):
    dim = X.shape[1]
    agents = np.random.randint(0, 2, (n_agents, dim))  # Binary initial population
    fitnesses = np.array([fitness(agent, X, y) for agent in agents])
    best_idx = np.argmax(fitnesses)
    best = agents[best_idx].copy()

    for _ in range(max_iter):
        proposals = np.array([agent ^ (np.random.rand(dim) < 0.1).astype(int) for agent in agents])
        proposal_fitnesses = np.array([fitness(prop, X, y) for prop in proposals])

        evaluations = np.zeros((n_agents, n_agents))
        for i in range(n_agents):
            for j in range(n_agents):
                if i != j:
                    evaluations[i][j] = proposal_fitnesses[j] - fitnesses[i]

        votes = np.zeros(n_agents)
        for i in range(n_agents):
            best_j = np.argmax(evaluations[i])
            votes[best_j] += 1

        best_proposal_idx = np.argmax(votes)
        best_proposal = proposals[best_proposal_idx]

        for i in range(n_agents):
            agents[i] = np.where(np.random.rand(dim) < alpha, best_proposal, agents[i])

        fitnesses = np.array([fitness(agent, X, y) for agent in agents])
        current_best_idx = np.argmax(fitnesses)
        if fitnesses[current_best_idx] > fitness(best, X, y):
            best = agents[current_best_idx].copy()

    return best

def run(X, y):
    start = time.time()
    best = CDMO(X, y)
    acc = fitness(best, X, y)
    return best, acc, np.sum(best), time.time() - start

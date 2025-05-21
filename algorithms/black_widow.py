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

def crossover(parent1, parent2):
    point = np.random.randint(1, len(parent1) - 1)
    child1 = np.concatenate((parent1[:point], parent2[point:]))
    child2 = np.concatenate((parent2[:point], parent1[point:]))
    return child1, child2

def mutate(child, mutation_rate=0.1):
    for i in range(len(child)):
        if np.random.rand() < mutation_rate:
            child[i] = 1 - child[i]
    return child

def BWO(X, y, n_agents=20, max_iter=50):
    dim = X.shape[1]
    population = np.random.randint(0, 2, (n_agents, dim))
    fitnesses = np.array([fitness(ind, X, y) for ind in population])
    best = population[np.argmax(fitnesses)].copy()

    for _ in range(max_iter):
        new_population = []
        for i in range(n_agents // 2):
            p1, p2 = population[np.random.randint(n_agents)], population[np.random.randint(n_agents)]
            c1, c2 = crossover(p1, p2)
            c1 = mutate(c1)
            c2 = mutate(c2)
            new_population.extend([c1, c2])

        all_population = np.array(new_population)
        all_fitnesses = np.array([fitness(ind, X, y) for ind in all_population])
        survivors_idx = np.argsort(-all_fitnesses)[:n_agents]
        population = all_population[survivors_idx]
        fitnesses = all_fitnesses[survivors_idx]
        if np.max(fitnesses) > fitness(best, X, y):
            best = population[np.argmax(fitnesses)].copy()

    return best

def run(X, y):
    start = time.time()
    best = BWO(X, y)
    acc = fitness(best, X, y)
    return best, acc, np.sum(best), time.time() - start

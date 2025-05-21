import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import time

def fitness(solution, X, y):
    if np.sum(solution) == 0:
        return 0
    X_selected = X[:, solution == 1]
    clf = RandomForestClassifier(n_estimators=100)
    return np.mean(cross_val_score(clf, X_selected, y, cv=5))

def binarize(position):
    sigmoid = 1 / (1 + np.exp(-position))
    binary = (sigmoid > 0.5).astype(int)
    if np.sum(binary) == 0:
        binary[np.random.randint(0, len(binary))] = 1
    return binary

def simulated_annealing(solution, X, y, initial_temp=1000, final_temp=0.01, alpha=0.95):
    temp = initial_temp
    current_solution = solution.copy()
    current_fitness = fitness(current_solution, X, y)

    while temp > final_temp:
        new_solution = current_solution + np.random.normal(0, 0.1, len(solution))
        new_solution = np.clip(new_solution, 0, 1)
        new_solution = binarize(new_solution)
        new_fitness = fitness(new_solution, X, y)

        if new_fitness > current_fitness or np.random.rand() < np.exp((new_fitness - current_fitness) / temp):
            current_solution = new_solution
            current_fitness = new_fitness

        temp *= alpha

    return current_solution

def crossover(parent1, parent2):
    point = np.random.randint(1, len(parent1) - 1)
    child1 = np.concatenate((parent1[:point], parent2[point:]))
    child2 = np.concatenate((parent2[:point], parent1[point:]))
    return child1, child2

def mutation(solution, mutation_rate=0.1):
    mutated = solution.copy()
    for i in range(len(mutated)):
        if np.random.rand() < mutation_rate:
            mutated[i] = 1 - mutated[i]
    return mutated

def hybrid_sa_ga(X, y, n_individuals=20, max_iter=50, mutation_rate=0.1):
    dim = X.shape[1]
    population = np.random.rand(n_individuals, dim)
    binaries = np.array([binarize(ind) for ind in population])
    fitnesses = np.array([fitness(ind, X, y) for ind in binaries])

    best_idx = np.argmax(fitnesses)
    best_solution = binaries[best_idx].copy()
    best_score = fitnesses[best_idx]

    for _ in range(max_iter):
        sorted_idx = np.argsort(fitnesses)[::-1]
        population = population[sorted_idx]
        binaries = binaries[sorted_idx]
        fitnesses = fitnesses[sorted_idx]

        new_population = []
        for i in range(0, n_individuals, 2):
            parent1, parent2 = binaries[i], binaries[i+1]
            child1, child2 = crossover(parent1, parent2)
            new_population.append(mutation(child1, mutation_rate))
            new_population.append(mutation(child2, mutation_rate))

        new_binaries = [simulated_annealing(ind, X, y) for ind in new_population]
        new_fitnesses = np.array([fitness(ind, X, y) for ind in new_binaries])
        
        binaries = np.array(new_binaries)
        fitnesses = new_fitnesses

        best_idx = np.argmax(fitnesses)
        if fitnesses[best_idx] > best_score:
            best_solution = binaries[best_idx].copy()
            best_score = fitnesses[best_idx]

    return best_solution

def run(X, y):
    start = time.time()
    best = hybrid_sa_ga(X, y)
    acc = fitness(best, X, y)
    return best, acc, np.sum(best), time.time() - start

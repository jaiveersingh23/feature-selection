import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import time

def fitness(solution, X, y, alpha=0.98):
    if np.sum(solution) == 0:
        return 0
    acc = np.mean(cross_val_score(KNeighborsClassifier(n_neighbors=5), X[:, solution == 1], y, cv=5))
    feature_ratio = np.sum(solution) / len(solution)
    return alpha * acc + (1 - alpha) * (1 - feature_ratio)


def binarize(position):
    return (position > 0.5).astype(int)

def heat_diffusion(agents, diffusion_rate=0.1):
    new_agents = agents.copy()
    for i in range(len(agents)):
        neighbors = np.random.choice(len(agents), size=3, replace=False)
        neighbor_avg = np.mean(agents[neighbors], axis=0)
        new_agents[i] += diffusion_rate * (neighbor_avg - agents[i])
    return np.clip(new_agents, 0, 1)

def cooling(agents, cooling_factor=0.99):
    return agents * cooling_factor

def THDOA(X, y, n_agents=20, max_iter=50):
    dim = X.shape[1]
    agents = np.random.rand(n_agents, dim)
    fitness_values = np.array([fitness(binarize(agent), X, y) for agent in agents])
    best_idx = np.argmax(fitness_values)
    best_agent = agents[best_idx].copy()
    best_fitness = fitness_values[best_idx]

    for _ in range(max_iter):
        agents = heat_diffusion(agents)
        agents = cooling(agents)
        fitness_values = np.array([fitness(binarize(agent), X, y) for agent in agents])
        current_best_idx = np.argmax(fitness_values)
        if fitness_values[current_best_idx] > best_fitness:
            best_fitness = fitness_values[current_best_idx]
            best_agent = agents[current_best_idx].copy()
    return binarize(best_agent)

def HOA(X, y, initial_best, n_agents=20, max_iter=50):
    dim = X.shape[1]
    horses = np.random.randint(0, 2, (n_agents, dim))
    horses[0] = initial_best  # Inject THDOA best as first horse

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
    
    # Phase 1: THDOA for high accuracy
    thdoa_best = THDOA(X, y)
    
    # Phase 2: HOA to reduce feature count while keeping accuracy
    hybrid_best = HOA(X, y, initial_best=thdoa_best)
    
    acc = fitness(hybrid_best, X, y)
    feature_count = np.sum(hybrid_best)
    elapsed = time.time() - start
    
    return hybrid_best, acc, feature_count, elapsed

import numpy as np
import time
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

# ---- Fitness ----
def fitness(solution, X, y, alpha=0.98):
    if np.sum(solution) == 0:
        return 0
    acc = np.mean(cross_val_score(KNeighborsClassifier(n_neighbors=5), X[:, solution == 1], y, cv=5))
    feature_ratio = np.sum(solution) / len(solution)
    return alpha * acc + (1 - alpha) * (1 - feature_ratio)

# ---- Binarization ----
def binarize(position):
    return (position > 0.5).astype(int)

# ---- THDOA ----
def heat_diffusion(agents, fitness_values, diffusion_rate=0.1):
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
        agents = heat_diffusion(agents, fitness_values)
        agents = cooling(agents)
        fitness_values = np.array([fitness(binarize(agent), X, y) for agent in agents])
        current_best_idx = np.argmax(fitness_values)
        if fitness_values[current_best_idx] > best_fitness:
            best_fitness = fitness_values[current_best_idx]
            best_agent = agents[current_best_idx].copy()

    return best_agent  # returns continuous best agent

# ---- IEHO Phase ----
def herd_movement(position, leader_factor):
    return np.clip(position + leader_factor * (np.random.rand(len(position)) - 0.5), 0, 1)

def cooperative_strategy(position, elite_mean, cooperation_factor):
    return np.clip(position + cooperation_factor * elite_mean, 0, 1)

def IEHO(X, y, initial_agents, n_agents=20, max_iter=50, elite_frac=0.2, leader_factor=0.3, cooperation_factor=0.2):
    dim = X.shape[1]
    agents = initial_agents.copy()
    binaries = np.array([binarize(agent) for agent in agents])
    fitness_values = np.array([fitness(bin_vec, X, y) for bin_vec in binaries])

    best_idx = np.argmax(fitness_values)
    best_agent = agents[best_idx].copy()
    best_fitness = fitness_values[best_idx]

    for _ in range(max_iter):
        sorted_idx = np.argsort(fitness_values)[::-1]
        agents = agents[sorted_idx]
        binaries = binaries[sorted_idx]
        fitness_values = fitness_values[sorted_idx]

        n_elite = int(elite_frac * n_agents)
        elite = agents[:n_elite]
        elite_mean = np.mean(elite, axis=0)

        for i in range(n_elite, n_agents):
            new_position = herd_movement(agents[i], leader_factor)
            if np.random.rand() < cooperation_factor:
                new_position = cooperative_strategy(new_position, elite_mean, cooperation_factor)

            new_binary = binarize(new_position)
            new_fitness = fitness(new_binary, X, y)

            agents[i] = new_position
            binaries[i] = new_binary
            fitness_values[i] = new_fitness

            if new_fitness > best_fitness:
                best_fitness = new_fitness
                best_agent = new_position.copy()

    return best_agent

# ---- Hybrid THDOA + IEHO ----
def run(X, y, thdoa_iters=50, ieho_iters=50):
    start = time.time()

    # Phase 1: THDOA
    thdoa_best = THDOA(X, y, max_iter=thdoa_iters)
    
    # Generate a population around THDOA best
    dim = X.shape[1]
    n_agents = 20
    initial_agents = np.tile(thdoa_best, (n_agents, 1)) + 0.05 * np.random.randn(n_agents, dim)
    initial_agents = np.clip(initial_agents, 0, 1)

    # Phase 2: IEHO refinement
    hybrid_best = IEHO(X, y, initial_agents, max_iter=ieho_iters)

    final_binary = binarize(hybrid_best)
    acc = fitness(final_binary, X, y)
    features = np.sum(final_binary)
    elapsed = time.time() - start

    return final_binary, acc, features, elapsed

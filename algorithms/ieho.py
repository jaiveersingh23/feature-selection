import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import time

def fitness(solution, X, y):
    """Evaluate the fitness (accuracy) of a binary feature subset using KNN."""
    if np.sum(solution) == 0:
        return 0
    X_selected = X[:, solution == 1]
    clf = KNeighborsClassifier(n_neighbors=5)
    return np.mean(cross_val_score(clf, X_selected, y, cv=5))

def binarize(position):
    """Convert continuous to binary using 0.5 threshold."""
    return (position > 0.5).astype(int)

def herd_movement(position, leader_factor):
    """Apply herd movement (exploration) by random perturbation."""
    return np.clip(position + leader_factor * (np.random.rand(len(position)) - 0.5), 0, 1)

def cooperative_strategy(position, elite_mean, cooperation_factor):
    """Refine position based on elite average (cooperative behavior)."""
    return np.clip(position + cooperation_factor * elite_mean, 0, 1)

def elephant_herd_optimization(X, y, n_agents=20, max_iter=50, elite_frac=0.2, leader_factor=0.3, cooperation_factor=0.2):
    """Main loop for Indian Elephant Herd Optimization Algorithm."""
    dim = X.shape[1]
    agents = np.random.rand(n_agents, dim)
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
            # Herd Movement
            new_position = herd_movement(agents[i], leader_factor)

            # Cooperative Strategy
            if np.random.rand() < cooperation_factor:
                new_position = cooperative_strategy(new_position, elite_mean, cooperation_factor)

            new_binary = binarize(new_position)
            new_fitness = fitness(new_binary, X, y)

            agents[i] = new_position
            binaries[i] = new_binary
            fitness_values[i] = new_fitness

            # Leader Following
            if new_fitness > best_fitness:
                best_fitness = new_fitness
                best_agent = new_position.copy()

        # Optional convergence
        if best_fitness > np.mean(fitness_values):
            break

    return best_agent

def run(X, y):
    """Run IEHO and return best solution, accuracy, feature count, and time taken."""
    start = time.time()
    best = elephant_herd_optimization(X, y)
    best_binary = binarize(best)
    acc = fitness(best_binary, X, y)
    return best, acc, np.sum(best_binary), time.time() - start

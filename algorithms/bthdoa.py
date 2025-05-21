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

def binary_heat_diffusion(agents, diffusion_rate=0.1):
    """Binary heat diffusion based on flipping bits with probability related to neighbors."""
    new_agents = agents.copy()
    n_agents, dim = agents.shape

    for i in range(n_agents):
        neighbors = np.random.choice(n_agents, size=3, replace=False)
        neighbor_avg = np.mean(agents[neighbors], axis=0)  # average bits (0.0 to 1.0)
        
        # Flip bits where the average suggests disagreement with current agent
        for d in range(dim):
            if np.random.rand() < diffusion_rate:
                if neighbor_avg[d] > 0.5:
                    new_agents[i][d] = 1
                elif neighbor_avg[d] < 0.5:
                    new_agents[i][d] = 0
                # else leave it unchanged (equal influence)

    return new_agents

def binary_cooling(agents, cooling_prob=0.01):
    """Apply random bit flips to simulate cooling."""
    mutation = np.random.rand(*agents.shape) < cooling_prob
    return agents ^ mutation.astype(int)

def thermodynamic_heat_diffusion_optimization_binary(X, y, n_agents=20, max_iter=50):
    """Binary version of Thermodynamic Heat Diffusion Optimization Algorithm."""
    dim = X.shape[1]
    agents = np.random.randint(0, 2, size=(n_agents, dim))  # Binary initialization
    fitness_values = np.array([fitness(agent, X, y) for agent in agents])

    best_idx = np.argmax(fitness_values)
    best_agent = agents[best_idx].copy()
    best_fitness = fitness_values[best_idx]

    for _ in range(max_iter):
        agents = binary_heat_diffusion(agents, diffusion_rate=0.1)
        agents = binary_cooling(agents, cooling_prob=0.01)
        fitness_values = np.array([fitness(agent, X, y) for agent in agents])

        current_best_idx = np.argmax(fitness_values)
        if fitness_values[current_best_idx] > best_fitness:
            best_fitness = fitness_values[current_best_idx]
            best_agent = agents[current_best_idx].copy()

    return best_agent

def run(X, y):
    """Run Binary THDOA and return best solution, accuracy, feature count, and time taken."""
    start = time.time()
    best = thermodynamic_heat_diffusion_optimization_binary(X, y)
    acc = fitness(best, X, y)
    return best, acc, np.sum(best), time.time() - start

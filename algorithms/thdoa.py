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

def heat_diffusion(agents, fitness_values, diffusion_rate=0.1):
    """Spread fitness (heat) among agents simulating thermodynamic diffusion."""
    new_agents = agents.copy()
    for i in range(len(agents)):
        neighbors = np.random.choice(len(agents), size=3, replace=False)
        neighbor_avg = np.mean(agents[neighbors], axis=0)
        new_agents[i] += diffusion_rate * (neighbor_avg - agents[i])
    return np.clip(new_agents, 0, 1)

def cooling(agents, cooling_factor=0.99):
    """Simulate system cooling over time."""
    return agents * cooling_factor

def thermodynamic_heat_diffusion_optimization(X, y, n_agents=20, max_iter=50):
    """Main loop for Thermodynamic Heat Diffusion Optimization Algorithm."""
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

    return best_agent

def run(X, y):
    """Run THDOA and return best solution, accuracy, feature count, and time taken."""
    start = time.time()
    best = thermodynamic_heat_diffusion_optimization(X, y)
    acc = fitness(binarize(best), X, y)
    return best, acc, np.sum(binarize(best)), time.time() - start

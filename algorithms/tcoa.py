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
    """Convert continuous values to binary using threshold 0.5."""
    return (position > 0.5).astype(int)

def energy_cascade(agent_energy, large_scale_factor=1.0, small_scale_factor=0.5):
    """Simulate energy cascade from large-scale (exploration) to small-scale (exploitation)."""
    large_scale_energy = agent_energy * large_scale_factor
    small_scale_energy = agent_energy * small_scale_factor
    energy_diff = large_scale_energy - small_scale_energy
    return large_scale_energy, small_scale_energy, energy_diff

def turbulent_move(agent, energy_diff):
    """Move agent using chaotic turbulence based on energy difference."""
    direction = np.random.randn(len(agent))  # Random vector (turbulent force)
    new_position = agent + energy_diff * direction
    return np.clip(new_position, 0, 1)

def turbulent_cascade_optimization(X, y, n_agents=20, max_iter=50):
    """Main TCOA optimizer loop."""
    dim = X.shape[1]
    positions = np.random.rand(n_agents, dim)

    fitness_values = np.array([fitness(binarize(agent), X, y) for agent in positions])
    total_fitness = np.sum(fitness_values)
    if total_fitness == 0:
        energies = np.ones(n_agents) / n_agents
    else:
        energies = fitness_values / total_fitness

    best_idx = np.argmax(fitness_values)
    best_agent = positions[best_idx].copy()
    best_fitness = fitness_values[best_idx]

    for _ in range(max_iter):
        for i in range(n_agents):
            _, _, energy_diff = energy_cascade(energies[i])
            new_position = turbulent_move(positions[i], energy_diff)
            binary_position = binarize(new_position)
            new_fitness = fitness(binary_position, X, y)

            if new_fitness > fitness_values[i]:
                positions[i] = new_position
                fitness_values[i] = new_fitness

                if new_fitness > best_fitness:
                    best_fitness = new_fitness
                    best_agent = new_position.copy()

        total_fitness = np.sum(fitness_values)
        energies = fitness_values / total_fitness if total_fitness != 0 else np.ones(n_agents) / n_agents

    return best_agent

def run(X, y):
    """Run the TCOA algorithm and return results."""
    start = time.time()
    best = turbulent_cascade_optimization(X, y)
    acc = fitness(binarize(best), X, y)
    return best, acc, np.sum(binarize(best)), time.time() - start

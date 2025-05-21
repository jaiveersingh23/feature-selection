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
    """Convert continuous vector to binary using 0.5 threshold."""
    return (position > 0.5).astype(int)

def gravitational_force(agent, best_agent, G=1.0):
    """Calculate gravitational attraction force from best agent."""
    distance = np.linalg.norm(agent - best_agent)
    if distance == 0:
        return np.zeros_like(agent)
    force_magnitude = G * np.sum(agent) * np.sum(best_agent) / (distance ** 2)
    direction = (best_agent - agent) / distance
    return force_magnitude * direction

def update_position(agent, force, learning_rate=0.1):
    """Update agent's position with gravitational influence."""
    new_position = agent + learning_rate * force
    return np.clip(new_position, 0, 1)

def gravitational_distortion_optimization(X, y, n_agents=20, max_iter=50):
    """Main optimization loop of GDOA."""
    dim = X.shape[1]
    positions = np.random.rand(n_agents, dim)
    fitness_values = np.array([fitness(binarize(agent), X, y) for agent in positions])

    best_idx = np.argmax(fitness_values)
    best_agent = positions[best_idx].copy()
    best_fitness = fitness_values[best_idx]

    for _ in range(max_iter):
        for i in range(n_agents):
            force = gravitational_force(positions[i], best_agent)
            new_position = update_position(positions[i], force)
            binary_position = binarize(new_position)
            new_fitness = fitness(binary_position, X, y)

            if new_fitness > fitness_values[i]:
                positions[i] = new_position
                fitness_values[i] = new_fitness

                if new_fitness > best_fitness:
                    best_fitness = new_fitness
                    best_agent = new_position.copy()

    return best_agent

def run(X, y):
    """Run GDOA and return best solution, accuracy, number of features, and time."""
    start = time.time()
    best = gravitational_distortion_optimization(X, y)
    acc = fitness(binarize(best), X, y)
    return best, acc, np.sum(binarize(best)), time.time() - start

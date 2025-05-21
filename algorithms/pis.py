import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import time

def fitness(solution, X, y):
    """Evaluate fitness (accuracy) of a binary feature subset using KNN."""
    if np.sum(solution) == 0:
        return 0
    X_selected = X[:, solution == 1]
    clf = KNeighborsClassifier(n_neighbors=5)
    return np.mean(cross_val_score(clf, X_selected, y, cv=5))

def binarize(position):
    """Convert continuous to binary using 0.5 threshold."""
    return (position > 0.5).astype(int)

def pictorial_distance(sol1, sol2):
    """Symbolic and geometric distance between two solutions."""
    return np.linalg.norm(sol1 - sol2) * abs(np.sum(sol1) - np.sum(sol2))

def pictorial_influence(agent, best_solution, repulsion_factor=0.5, max_distance=10):
    """Adjust agent based on distance to best (attract or repel)."""
    distance = pictorial_distance(agent, best_solution)
    # Clamp the distance to avoid excessive influence
    if distance > max_distance:
        distance = max_distance
    
    if distance < 0.5:
        return agent + (1 - distance) * (best_solution - agent)  # Attraction
    else:
        return agent - repulsion_factor * (distance - 0.5) * (agent - best_solution)  # Repulsion

    # Ensure that agents stay in the [0, 1] range
    return np.clip(agent, 0, 1)

"""
def pictorial_influence(agent, best_solution, repulsion_factor=0.5):
    #Adjust agent based on distance to best (attract or repel).
    distance = pictorial_distance(agent, best_solution)
    if distance < 0.5:
        return agent + (1 - distance) * (best_solution - agent)  # Attraction
    else:
        return agent - repulsion_factor * (distance - 0.5) * (agent - best_solution)  # Repulsion
"""

def symbolic_refinement(agents, best_solution):
    """Update all agents via pictorial influence from the best."""
    return np.array([pictorial_influence(agent, best_solution) for agent in agents])

def pictorial_influence_search(X, y, n_agents=20, max_iter=50):
    """Main loop of Pictorial Influence Search Algorithm."""
    dim = X.shape[1]
    agents = np.random.rand(n_agents, dim)
    fitness_values = np.array([fitness(binarize(agent), X, y) for agent in agents])

    best_idx = np.argmax(fitness_values)
    best_solution = agents[best_idx].copy()
    best_fitness = fitness_values[best_idx]

    for _ in range(max_iter):
        agents = symbolic_refinement(agents, best_solution)
        fitness_values = np.array([fitness(binarize(agent), X, y) for agent in agents])

        current_best_idx = np.argmax(fitness_values)
        if fitness_values[current_best_idx] > best_fitness:
            best_fitness = fitness_values[current_best_idx]
            best_solution = agents[current_best_idx].copy()

    return best_solution

def run(X, y):
    """Run PIS and return best solution, accuracy, feature count, and time taken."""
    start = time.time()
    best = pictorial_influence_search(X, y)
    acc = fitness(binarize(best), X, y)
    return best, acc, np.sum(binarize(best)), time.time() - start

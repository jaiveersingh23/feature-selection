import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import time

def fitness(solution, X, y):
    """Evaluate fitness (accuracy) of a solution using a classifier."""
    if np.sum(solution) == 0:
        return 0
    X_selected = X[:, solution == 1]
    clf = KNeighborsClassifier(n_neighbors=5)
    score = np.mean(cross_val_score(clf, X_selected, y, cv=5))
    return score

def ensure_at_least_one_feature(solution):
    """Ensure at least one feature is selected."""
    if np.sum(solution) == 0:
        idx = np.random.randint(0, len(solution))
        solution[idx] = 1
    return solution

def binarize(position):
    """Convert a continuous position to binary solution using sigmoid function."""
    sigmoid = 1 / (1 + np.exp(-position))
    binary = (sigmoid > 0.5).astype(int)
    return ensure_at_least_one_feature(binary)

def compute_avoidance(current, others, radius=0.3):
    """Repulsion from other agents based on distance and a defined radius."""
    repulsion = np.zeros_like(current)
    for other in others:
        distance = np.linalg.norm(current - other)
        if 0 < distance < radius:
            repulsion += (current - other) / (distance**2)
    return repulsion

def FNO(X, y, n_agents=20, max_iter=50, α=0.5, β=0.4, γ=0.1):
    """Footpath Navigation Optimization algorithm."""
    dim = X.shape[1]
    population = np.random.rand(n_agents, dim)
    binaries = np.array([binarize(ind) for ind in population])
    fitnesses = np.array([fitness(b, X, y) for b in binaries])
    best_idx = np.argmax(fitnesses)
    best_binary = binaries[best_idx].copy()
    best_position = population[best_idx].copy()

    for _ in range(max_iter):
        for i in range(n_agents):
            others = np.delete(population, i, axis=0)
            avoidance = compute_avoidance(population[i], others)
            noise = np.random.uniform(-1, 1, dim)

            new_position = (
                population[i] 
                + α * (best_position - population[i]) 
                + β * avoidance 
                + γ * noise
            )
            new_position = np.clip(new_position, 0, 1)
            new_binary = binarize(new_position)
            new_fit = fitness(new_binary, X, y)

            if new_fit > fitnesses[i]:
                population[i] = new_position
                binaries[i] = new_binary
                fitnesses[i] = new_fit

                if new_fit > fitness(best_binary, X, y):
                    best_binary = new_binary.copy()
                    best_position = new_position.copy()

    return best_binary

def run(X, y):
    """Run the FNO algorithm and return results."""
    start = time.time()
    best = FNO(X, y)
    acc = fitness(best, X, y)
    return best, acc, np.sum(best), time.time() - start

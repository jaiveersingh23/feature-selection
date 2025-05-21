import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import time

def binarize(position):
    """Convert continuous vector to binary using sigmoid transformation."""
    sigmoid = 1 / (1 + np.exp(-position))
    binary = (sigmoid > 0.5).astype(int)
    
    # Ensure at least one feature is selected
    if np.sum(binary) == 0:
        binary[np.random.randint(len(binary))] = 1
        
    return binary


def fitness(solution, X, y):
    """Evaluate the fitness (accuracy) of a binary feature subset using RandomForest."""
    if np.sum(solution) == 0:
        return 0  # Avoid zero-feature case
    X_selected = X[:, solution == 1]
    clf = RandomForestClassifier()
    return np.mean(cross_val_score(clf, X_selected, y, cv=5))

def influence(i, j, s_i, s_j, f_i, f_j, f_max, f_min):
    """Calculate the influence of agent i on agent j."""
    distance = np.linalg.norm(s_i - s_j)
    if f_max == f_min:  # Avoid division by zero
        return 0
    influence_value = 1 / (1 + distance) * (f_i - f_j) / (f_max - f_min)
    return influence_value

def synchronize(agents, fitness_values, alpha=0.1):
    """Update agents based on social synchronization."""
    n_agents, dim = agents.shape
    new_agents = agents.copy()

    f_max = np.max(fitness_values)
    f_min = np.min(fitness_values)

    for i in range(n_agents):
        s_i = agents[i]
        f_i = fitness_values[i]
        influence_sum = np.zeros(dim)
        for j in range(n_agents):
            if i != j:
                s_j = agents[j]
                f_j = fitness_values[j]
                infl = influence(i, j, s_i, s_j, f_i, f_j, f_max, f_min)
                influence_sum += infl * (s_j - s_i)
        new_agents[i] = s_i + alpha * influence_sum

    return np.clip(new_agents, 0, 1)

def social_synchronization_optimization(X, y, n_agents=20, max_iter=50):
    """Main optimization loop for SoSO."""
    dim = X.shape[1]
    agents = np.random.rand(n_agents, dim)
    fitness_values = np.array([fitness(binarize(agent), X, y) for agent in agents])

    best_idx = np.argmax(fitness_values)
    best_agent = agents[best_idx].copy()
    best_fitness = fitness_values[best_idx]

    for _ in range(max_iter):
        agents = synchronize(agents, fitness_values)
        fitness_values = np.array([fitness(binarize(agent), X, y) for agent in agents])

        current_best_idx = np.argmax(fitness_values)
        if fitness_values[current_best_idx] > best_fitness:
            best_fitness = fitness_values[current_best_idx]
            best_agent = agents[current_best_idx].copy()

    return binarize(best_agent), best_fitness

def run(X, y):
    """Run SoSO and return best solution, accuracy, feature count, and time taken."""
    start = time.time()
    best, acc = social_synchronization_optimization(X, y)
    return best, acc, np.sum(best), time.time() - start

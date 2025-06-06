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

def binarize(weights, threshold=0.5):
    return (weights > threshold).astype(int)

def DMFS(X, y, n_agents=30, max_iter=60, dopamine_decay=0.9, learning_rate=0.1, noise_scale=0.02):
    dim = X.shape[1]
    agents = np.random.rand(n_agents, dim)
    fitness_values = np.array([fitness(binarize(agent), X, y) for agent in agents])
    best_idx = np.argmax(fitness_values)
    best_agent = agents[best_idx].copy()
    best_fitness = fitness_values[best_idx]

    dopamine = 0.5  # initial dopamine level

    for iteration in range(max_iter):
        prev_best_fitness = best_fitness

        for i in range(n_agents):
            weights = agents[i]

            # Update weights modulated by dopamine signal
            selected_features = binarize(best_agent)
            # Reinforce selected features
            weights += learning_rate * dopamine * selected_features
            # Decay unselected features
            weights -= learning_rate * dopamine * (1 - selected_features)

            # Add exploration noise
            weights += np.random.normal(0, noise_scale, dim)

            # Keep weights in [0,1]
            agents[i] = np.clip(weights, 0, 1)

        # Evaluate updated agents
        fitness_values = np.array([fitness(binarize(agent), X, y) for agent in agents])
        current_best_idx = np.argmax(fitness_values)

        if fitness_values[current_best_idx] > best_fitness:
            best_fitness = fitness_values[current_best_idx]
            best_agent = agents[current_best_idx].copy()

        # Dopamine update (reward signal = fitness improvement)
        reward = max(0, best_fitness - prev_best_fitness)
        dopamine = dopamine * dopamine_decay + reward * (1 - dopamine_decay)

    return binarize(best_agent)

def run(X, y):
    start = time.time()

    best_solution = DMFS(X, y)

    acc = fitness(best_solution, X, y)
    feature_count = np.sum(best_solution)
    elapsed = time.time() - start

    return best_solution, acc, feature_count, elapsed

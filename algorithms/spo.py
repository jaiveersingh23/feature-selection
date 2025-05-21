import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import time

def fitness(solution, X, y):
    if np.sum(solution) == 0:
        return 0  # Defensive fallback (should not occur with correction below)
    X_selected = X[:, solution == 1]
    clf = RandomForestClassifier(n_estimators=100)
    return np.mean(cross_val_score(clf, X_selected, y, cv=5))

def ensure_at_least_one_feature(binary):
    if np.sum(binary) == 0:
        binary[np.random.randint(0, len(binary))] = 1
    return binary

def binarize(position):
    binary = (position > 0.5).astype(int)
    return ensure_at_least_one_feature(binary)

def wave_propagation(agent, best_agent, wave_speed, max_distance):
    direction = best_agent - agent
    distance = np.linalg.norm(direction)
    if distance > max_distance:
        return agent + wave_speed * direction / distance
    else:
        return best_agent

def reflection(agent, lower_bound, upper_bound):
    return np.clip(agent, lower_bound, upper_bound)

def sound_wave_optimization(X, y, n_agents=20, max_iter=50, wave_speed=0.1, max_distance=1.0, lower_bound=0, upper_bound=1):
    dim = X.shape[1]
    positions = np.random.rand(n_agents, dim)
    fitness_values = np.array([fitness(binarize(agent), X, y) for agent in positions])

    best_idx = np.argmax(fitness_values)
    best_agent = positions[best_idx].copy()
    best_fitness = fitness_values[best_idx]

    for t in range(max_iter):
        for i in range(n_agents):
            new_position = wave_propagation(positions[i], best_agent, wave_speed, max_distance)
            new_position = reflection(new_position, lower_bound, upper_bound)

            binary = binarize(new_position)
            fit = fitness(binary, X, y)

            fitness_values[i] = fit

            if fit > best_fitness:
                best_fitness = fit
                best_agent = new_position

    return best_agent

def run(X, y):
    start = time.time()
    best = sound_wave_optimization(X, y)
    acc = fitness(best, X, y)
    return best, acc, np.sum(best), time.time() - start

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import time

def fitness(solution, X, y):
    if np.sum(solution) == 0:
        return 0
    X_selected = X[:, solution == 1]
    clf = KNeighborsClassifier(n_neighbors=5)
    return np.mean(cross_val_score(clf, X_selected, y, cv=5))

def binarize(position):
    prob = 1 / (1 + np.exp(-position))
    binary = (prob > 0.5).astype(int)
    if np.sum(binary) == 0:
        binary[np.random.randint(len(binary))] = 1
    return binary

def gravitational_force(xi, xj, mi, mj, G):
    distance = np.linalg.norm(xj - xi) + 1e-9
    return G * (mi * mj / (distance**2)) * (xj - xi) / distance

def GFO(X, y, n_agents=20, max_iter=50):
    dim = X.shape[1]
    positions = np.random.rand(n_agents, dim)
    velocities = np.zeros((n_agents, dim))
    G = 0.01

    binaries = np.array([binarize(p) for p in positions])
    fitnesses = np.array([fitness(b, X, y) for b in binaries])
    masses = fitnesses / (np.sum(fitnesses) + 1e-9)

    best_idx = np.argmax(fitnesses)
    best_binary = binaries[best_idx].copy()
    best_position = positions[best_idx].copy()

    for _ in range(max_iter):
        for i in range(n_agents):
            total_force = np.zeros(dim)
            for j in range(n_agents):
                if i != j:
                    force = gravitational_force(positions[i], positions[j], masses[i], masses[j], G)
                    total_force += force

                    # Fusion: average positions if too close
                    if np.linalg.norm(positions[i] - positions[j]) < 0.1:
                        positions[i] = (masses[i] * positions[i] + masses[j] * positions[j]) / (masses[i] + masses[j] + 1e-9)

            velocities[i] = 0.5 * velocities[i] + total_force
            positions[i] += velocities[i]
            positions[i] = np.clip(positions[i], 0, 1)

        binaries = np.array([binarize(p) for p in positions])
        fitnesses = np.array([fitness(b, X, y) for b in binaries])
        masses = fitnesses / (np.sum(fitnesses) + 1e-9)

        current_best_idx = np.argmax(fitnesses)
        if fitnesses[current_best_idx] > fitness(best_binary, X, y):
            best_binary = binaries[current_best_idx].copy()
            best_position = positions[current_best_idx].copy()

        G *= 1.01  # Gradual energy increase

    return best_binary

def run(X, y):
    start = time.time()
    best = GFO(X, y)
    acc = fitness(best, X, y)
    return best, acc, np.sum(best), time.time() - start

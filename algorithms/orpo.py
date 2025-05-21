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
    return (position > 0.5).astype(int)

def reflect(ray, angle_of_incidence):
    """Reflect ray by perturbing its direction based on angle (used as strength)."""
    perturbation = (np.random.rand(*ray.shape) - 0.5) * angle_of_incidence
    return ray + perturbation

def refract(ray, n1, n2, angle_of_incidence):
    """Refract ray by perturbing its direction based on Snell's Law."""
    try:
        angle_of_refraction = np.arcsin(n1 * np.sin(angle_of_incidence) / n2)
    except ValueError:
        angle_of_refraction = angle_of_incidence
    perturbation = (np.random.rand(*ray.shape) - 0.5) * angle_of_refraction
    return ray + perturbation


def ray_optimization(X, y, n_agents=20, max_iter=50, lower_bound=0, upper_bound=1):
    dim = X.shape[1]
    positions = np.random.rand(n_agents, dim)
    fitness_values = np.array([fitness(binarize(agent), X, y) for agent in positions])

    best_idx = np.argmax(fitness_values)
    best_agent = positions[best_idx].copy()
    best_fitness = fitness_values[best_idx]

    n1, n2 = 1.0, 1.5

    for _ in range(max_iter):
        for i in range(n_agents):
            angle = np.random.rand() * np.pi / 2

            if np.random.rand() < 0.5:
                new_position = reflect(positions[i], angle)
            else:
                new_position = refract(positions[i], n1, n2, angle)

            new_position = np.clip(new_position, lower_bound, upper_bound)
            binary = binarize(new_position)
            fit = fitness(binary, X, y)

            fitness_values[i] = fit

            if fit > best_fitness:
                best_fitness = fit
                best_agent = new_position

    return best_agent

def run(X, y):
    start = time.time()
    best = ray_optimization(X, y)
    acc = fitness(best, X, y)
    return best, acc, np.sum(best), time.time() - start

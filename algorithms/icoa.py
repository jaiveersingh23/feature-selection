import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import time

def fitness(solution, X, y):
    """Evaluate the accuracy of the selected features."""
    X_selected = X[:, solution == 1]
    clf = KNeighborsClassifier(n_neighbors=5)
    return np.mean(cross_val_score(clf, X_selected, y, cv=5))

def ensure_at_least_one_feature(solution):
    """Ensure that a solution has at least one feature selected."""
    if np.sum(solution) == 0:
        idx = np.random.randint(0, len(solution))
        solution[idx] = 1
    return solution

def binarize_position(position):
    """Convert continuous position to binary using sigmoid + threshold."""
    sigmoid = 1 / (1 + np.exp(-position))
    binary = (sigmoid >= 0.5).astype(int)
    return ensure_at_least_one_feature(binary)

def update_position(current, A, C, attacker, barrier, chaser, driver):
    D1 = np.abs(C[0] * attacker - A[0] * current)
    D2 = np.abs(C[1] * barrier  - A[1] * current)
    D3 = np.abs(C[2] * chaser   - A[2] * current)
    D4 = np.abs(C[3] * driver   - A[3] * current)

    X1 = attacker - A[0] * D1
    X2 = barrier  - A[1] * D2
    X3 = chaser   - A[2] * D3
    X4 = driver   - A[3] * D4

    return np.clip((X1 + X2 + X3 + X4) / 4.0, 0, 1)

def ICOA(X, y, n_agents=20, max_iter=50):
    dim = X.shape[1]
    positions = np.random.uniform(low=-1, high=1, size=(n_agents, dim))
    binaries = np.array([binarize_position(pos) for pos in positions])
    fitnesses = np.array([fitness(sol, X, y) for sol in binaries])

    # Sort by fitness (descending)
    sorted_idx = np.argsort(fitnesses)[::-1]
    binaries = binaries[sorted_idx]
    positions = positions[sorted_idx]
    fitnesses = fitnesses[sorted_idx]

    # Select top 4
    attacker = positions[0]
    barrier  = positions[1]
    chaser   = positions[2]
    driver   = positions[3]

    for _ in range(max_iter):
        for i in range(n_agents):
            r1 = np.random.rand(4)
            r2 = np.random.rand(4)
            A  = 2 * r1 - 1
            C  = 2 * r2

            new_pos = update_position(positions[i], A, C, attacker, barrier, chaser, driver)
            new_bin = binarize_position(new_pos)

            new_fitness = fitness(new_bin, X, y)
            if new_fitness > fitnesses[i]:
                positions[i] = new_pos
                binaries[i]  = new_bin
                fitnesses[i] = new_fitness

        # Re-sort
        sorted_idx = np.argsort(fitnesses)[::-1]
        positions = positions[sorted_idx]
        binaries = binaries[sorted_idx]
        fitnesses = fitnesses[sorted_idx]

        attacker = positions[0]
        barrier  = positions[1]
        chaser   = positions[2]
        driver   = positions[3]

    return binaries[0]

def run(X, y):
    start = time.time()
    best_solution = ICOA(X, y)
    acc = fitness(best_solution, X, y)
    selected_features = np.sum(best_solution)
    end = time.time()
    return best_solution, acc, selected_features, end - start

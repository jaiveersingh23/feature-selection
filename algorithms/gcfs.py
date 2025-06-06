import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import mutual_info_classif
import time

# ---------- FITNESS FUNCTION ----------
def fitness(solution, X, y, alpha=0.98):
    if np.sum(solution) == 0:
        return 0
    acc = np.mean(cross_val_score(KNeighborsClassifier(n_neighbors=5), X[:, solution == 1], y, cv=5))
    feature_ratio = np.sum(solution) / len(solution)
    return alpha * acc + (1 - alpha) * (1 - feature_ratio)

# ---------- BINARIZATION ----------
def binarize(position):
    return (position > 0.5).astype(int)

# ---------- GCFS v2 MAIN OPTIMIZER ----------
def GCFS_v2(X, y, n_agents=20, max_iter=50):
    dim = X.shape[1]
    agents = np.random.rand(n_agents, dim)
    fitness_values = np.array([fitness(binarize(agent), X, y) for agent in agents])
    best_idx = np.argmax(fitness_values)
    best_agent = agents[best_idx].copy()
    best_fitness = fitness_values[best_idx]

    # Initial masses by mutual info normalized
    masses = mutual_info_classif(X, y)
    masses = (masses - masses.min()) / (masses.max() - masses.min() + 1e-8)

    G_init = 1.0
    eta_init = 0.05
    epsilon = 1e-6

    # Precompute feature correlation matrix for repulsion (Pearson corr)
    corr_matrix = np.corrcoef(X, rowvar=False)
    corr_matrix = np.nan_to_num(corr_matrix)  # replace NaNs with 0
    # Make sure diagonal zero to avoid self repulsion
    np.fill_diagonal(corr_matrix, 0)

    for iter_ in range(max_iter):
        G = G_init * (1 - iter_ / max_iter)
        eta = eta_init * (1 - iter_ / max_iter)

        for i in range(n_agents):
            for d in range(dim):
                grav_force = 0
                repul_force = 0

                for j in range(dim):
                    if j == d:
                        continue
                    dist = abs(agents[i, d] - agents[i, j])
                    # Gravitational attraction (pull together)
                    grav = G * (masses[d] * masses[j]) / (dist ** 2 + epsilon)
                    direction = np.sign(agents[i, j] - agents[i, d])
                    grav_force += grav * direction

                    # Repulsion proportional to correlation (push away if highly correlated)
                    repulsion_strength = corr_matrix[d, j]  # range -1 to 1, positive = repel
                    repul_force += repulsion_strength * (-direction) * (1 / (dist + epsilon))

                # Total force: attraction + repulsion
                total_force = grav_force + repul_force
                agents[i, d] += eta * total_force
                agents[i, d] = np.clip(agents[i, d], 0, 1)

            # Dynamic mass update after agent update based on feature contribution
            binarized = binarize(agents[i])
            for f in range(dim):
                if binarized[f] == 1:
                    # Increase mass slightly if selected feature improves fitness
                    temp_agent = binarized.copy()
                    temp_agent[f] = 0
                    fitness_without_f = fitness(temp_agent, X, y)
                    fitness_with_f = fitness(binarized, X, y)
                    improvement = fitness_with_f - fitness_without_f
                    masses[f] = masses[f] + 0.1 * improvement
                else:
                    # Decay mass if not selected
                    masses[f] *= 0.95

        # Normalize masses to [0,1]
        masses = np.clip(masses, 0, None)
        masses = (masses - masses.min()) / (masses.max() - masses.min() + 1e-8)

        # Evaluate fitness of all agents
        fitness_values = np.array([fitness(binarize(agent), X, y) for agent in agents])
        current_best_idx = np.argmax(fitness_values)
        if fitness_values[current_best_idx] > best_fitness:
            best_fitness = fitness_values[current_best_idx]
            best_agent = agents[current_best_idx].copy()

    return binarize(best_agent)

# ---------- RUN FUNCTION ----------
def run(X, y):
    start = time.time()

    gcfs_best = GCFS_v2(X, y)

    acc = fitness(gcfs_best, X, y)
    feature_count = np.sum(gcfs_best)
    elapsed = time.time() - start

    return gcfs_best, acc, feature_count, elapsed

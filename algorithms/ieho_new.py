import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import mutual_info_classif
import time

# --------- FITNESS FUNCTION (Error + Feature Ratio) ----------
def fitness(solution, X, y, beta=0.9):
    if np.sum(solution) == 0:
        return 1.0  # High error if no features selected
    X_sel = X[:, solution == 1]
    clf = KNeighborsClassifier(n_neighbors=5)
    err = 1 - np.mean(cross_val_score(clf, X_sel, y, cv=5))
    feature_ratio = np.sum(solution) / len(solution)
    return beta * err + (1 - beta) * feature_ratio

# --------- BINARIZATION ----------
def binarize(position):
    return (position > 0.5).astype(int)

# --------- HERD MOVEMENT ----------
def herd_movement(position, leader_factor):
    """Apply herd movement (exploration) by random perturbation."""
    return np.clip(position + leader_factor * (np.random.rand(len(position)) - 0.5), 0, 1)

# --------- COOPERATIVE STRATEGY ----------
def cooperative_strategy(position, elite_mean, cooperation_factor):
    """Refine position based on elite average (cooperative behavior)."""
    return np.clip(position + cooperation_factor * elite_mean, 0, 1)

# --------- PIECEWISE INITIALIZATION (SU-based) ----------
def piecewise_init(X, y, n_agents):
    SU = mutual_info_classif(X, y)
    feature_ranks = np.argsort(-SU)
    D = X.shape[1]
    M = int(0.1 * D)
    agents = np.zeros((n_agents, D))
    for i in range(n_agents):
        L = max(round((D / n_agents) * (i + 1)), M)
        for d in range(D):
            if d < M:
                agents[i, feature_ranks[d]] = 0.4 * np.random.rand() + 0.6
            elif d < L:
                agents[i, feature_ranks[d]] = np.random.rand()
    return np.clip(agents, 0, 1), SU

# --------- COMPREHENSIVE SCORE + REDUCTION ----------
def comprehensive_score(SU, freq, lam=0.5):
    freq_norm = freq / np.max(freq) if np.max(freq) > 0 else freq
    su_norm = SU / np.max(SU) if np.max(SU) > 0 else SU
    return lam * freq_norm + (1 - lam) * su_norm

def reduce_feature_space(SU, freq, best_bin_agent, alpha):
    scores = comprehensive_score(SU, freq)
    D = len(SU)
    top_k = int(alpha * D)
    top_indices = np.argsort(-scores)[:top_k]
    selected = np.where(best_bin_agent == 1)[0]
    new_space = np.union1d(top_indices, selected)
    return np.array(sorted(new_space))

# --------- IEHO ----------
def IEHO(X, y, n_agents=20, max_iter=50, elite_frac=0.2, leader_factor=0.3, cooperation_factor=0.2):
    original_dim = X.shape[1]  # Track original dim
    selected_indices_global = np.arange(original_dim)  # Track selected indices globally

    agents, SU = piecewise_init(X, y, n_agents)
    reduced_indices = np.arange(original_dim)

    bin_agents = np.array([binarize(a) for a in agents])
    fitness_vals = np.array([fitness(b, X, y) for b in bin_agents])
    best_idx = np.argmin(fitness_vals)
    best_agent = agents[best_idx].copy()
    best_bin = binarize(best_agent)
    best_fitness = fitness_vals[best_idx]

    for t in range(max_iter):
        # Sort agents by fitness
        sorted_idx = np.argsort(fitness_vals)
        agents = agents[sorted_idx]
        bin_agents = bin_agents[sorted_idx]
        fitness_vals = fitness_vals[sorted_idx]

        # Compute elite
        n_elite = int(elite_frac * n_agents)
        elite = agents[:n_elite]
        elite_mean = np.mean(elite, axis=0)

        for i in range(n_elite, n_agents):
            # Herd Movement
            new_position = herd_movement(agents[i], leader_factor)

            # Cooperative Strategy
            if np.random.rand() < cooperation_factor:
                new_position = cooperative_strategy(new_position, elite_mean, cooperation_factor)

            new_binary = binarize(new_position)
            new_fitness = fitness(new_binary, X, y)

            agents[i] = new_position
            bin_agents[i] = new_binary
            fitness_vals[i] = new_fitness

            # Update global best
            if new_fitness < best_fitness:
                best_fitness = new_fitness
                best_agent = new_position.copy()
                best_bin = new_binary.copy()

        # Feature reduction every T/5 iterations
        if (t + 1) % (max_iter // 5) == 0:
            freq = np.sum(bin_agents, axis=0)
            D = X.shape[1]
            if D <= 1000:
                alpha = 0.5 + (1 - 0.5) * np.random.rand()
            elif D <= 5000:
                alpha = 0.1 + (0.5 - 0.1) * np.random.rand()
            else:
                alpha = 0.05 + (0.1 - 0.05) * np.random.rand()
            selected = binarize(best_agent)
            reduced = reduce_feature_space(SU, freq, selected, alpha)

            # Update all components
            X = X[:, reduced]
            SU = SU[reduced]
            agents = agents[:, reduced]
            best_agent = best_agent[reduced]
            reduced_indices = reduced_indices[reduced]

            # Track reduction globally!
            selected_indices_global = selected_indices_global[reduced_indices]
            best_bin = binarize(best_agent)

    return binarize(best_agent), selected_indices_global


# --------- MAIN RUNNER ----------
def run(X, y):
    start = time.time()

    # Run IEHO
    ieho_best, selected_indices_global = IEHO(X, y)

    # Restore full length vector
    full_best = np.zeros(X.shape[1], dtype=int)
    full_best[selected_indices_global] = ieho_best

    final_fitness = fitness(full_best, X, y)
    acc = 1 - final_fitness  # Approximate accuracy
    feature_count = np.sum(full_best)
    elapsed = time.time() - start

    return full_best, acc, feature_count, elapsed

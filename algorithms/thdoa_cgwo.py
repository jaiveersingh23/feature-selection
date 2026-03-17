import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import mutual_info_classif
import time

# --------- FITNESS FUNCTION ----------
def fitness(solution, X, y, beta=0.9):
    if np.sum(solution) == 0:
        return 1.0
    X_sel = X[:, solution == 1]
    clf = KNeighborsClassifier(n_neighbors=5)
    err = 1 - np.mean(cross_val_score(clf, X_sel, y, cv=5))
    feature_ratio = np.sum(solution) / len(solution)
    return beta * err + (1 - beta) * feature_ratio

# --------- BINARIZATION ----------
def binarize(position):
    return (position > 0.5).astype(int)

# --------- HEAT DIFFUSION ----------
def heat_diffusion(agents, diffusion_rate=0.1):
    new_agents = agents.copy()
    for i in range(len(agents)):
        neighbors = np.random.choice(len(agents), size=3, replace=False)
        neighbor_avg = np.mean(agents[neighbors], axis=0)
        new_agents[i] += diffusion_rate * (neighbor_avg - agents[i])
    return np.clip(new_agents, 0, 1)

# --------- COOLING ----------
def cooling(agents, cooling_factor=0.99):
    return agents * cooling_factor

# --------- PIECEWISE INITIALIZATION ----------
def piecewise_init(X, y, n_agents):
    SU = mutual_info_classif(X, y)
    feature_ranks = np.argsort(-SU)
    D = X.shape[1]
    M = max(1, int(0.1 * D))
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

# --------- THDOA ----------
def THDOA(X, y, n_agents=20, max_iter=50):
    dim = X.shape[1]
    agents, SU = piecewise_init(X, y, n_agents)
    reduced_indices = np.arange(dim)

    bin_agents = np.array([binarize(a) for a in agents])
    fitness_vals = np.array([fitness(b, X, y) for b in bin_agents])
    best_idx = np.argmin(fitness_vals)
    best_agent = agents[best_idx].copy()
    best_bin = binarize(best_agent)
    best_fitness = fitness_vals[best_idx]

    for t in range(max_iter):
        agents = heat_diffusion(agents)
        agents = cooling(agents)
        bin_agents = np.array([binarize(a) for a in agents])
        fitness_vals = np.array([fitness(b, X, y) for b in bin_agents])
        curr_best_idx = np.argmin(fitness_vals)
        if fitness_vals[curr_best_idx] < best_fitness:
            best_fitness = fitness_vals[curr_best_idx]
            best_agent = agents[curr_best_idx].copy()
            best_bin = binarize(best_agent)

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

            X = X[:, reduced]
            SU = SU[reduced]
            agents = agents[:, reduced]
            best_agent = best_agent[reduced]
            reduced_indices = reduced_indices[reduced]
            best_bin = binarize(best_agent)

    return binarize(best_agent), reduced_indices, SU

# --------- CGWO (Chaotic Grey Wolf Optimizer) ----------
def GWO(X, y, initial_best, n_agents=30, max_iter=50):
    dim = X.shape[1]

    # --- Chebyshev chaotic map (from paper) ---
    def chebyshev_map(x):
        return np.cos(np.arccos(x))

    # Initial chaotic number (paper uses ~0.7)
    chaos = 0.7

    # Initialize wolves
    wolves = np.random.rand(n_agents, dim)
    wolves[0] = initial_best.copy()

    fitnesses = np.array([fitness(binarize(w), X, y) for w in wolves])

    for t in range(max_iter):

        # Sort wolves → alpha, beta, delta
        sorted_idx = np.argsort(fitnesses)
        alpha = wolves[sorted_idx[0]].copy()
        beta = wolves[sorted_idx[1]].copy()
        delta = wolves[sorted_idx[2]].copy()

        # --- Update chaotic sequence ---
        chaos = chebyshev_map(chaos)

        # --- Chaotic control parameter a ---
        a = 2 * chaos  # replaces linear decay

        for i in range(n_agents):

            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)

            A1 = 2 * a * r1 - a
            C1 = 2 * r2
            D_alpha = np.abs(C1 * alpha - wolves[i])
            X1 = alpha - A1 * D_alpha

            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)

            A2 = 2 * a * r1 - a
            C2 = 2 * r2
            D_beta = np.abs(C2 * beta - wolves[i])
            X2 = beta - A2 * D_beta

            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)

            A3 = 2 * a * r1 - a
            C3 = 2 * r2
            D_delta = np.abs(C3 * delta - wolves[i])
            X3 = delta - A3 * D_delta

            wolves[i] = np.clip((X1 + X2 + X3) / 3.0, 0, 1)

        # Recalculate fitness
        fitnesses = np.array([fitness(binarize(w), X, y) for w in wolves])

        # --- Replace worst wolf with best (paper step) ---
        worst = np.argmax(fitnesses)
        best = np.argmin(fitnesses)
        wolves[worst] = wolves[best].copy()
        fitnesses[worst] = fitnesses[best]

    best_idx = np.argmin(fitnesses)
    return binarize(wolves[best_idx])



# --------- FINAL PRUNING: Rank-based + Adaptive Greedy ----------
def final_pruning(X, y, selected_indices, best_bin, SU, base_acc=None, tol=0.001):
    X_selected = X[:, selected_indices]
    curr_features = np.where(best_bin == 1)[0]

    if base_acc is None:
        clf = KNeighborsClassifier(n_neighbors=5)
        base_acc = np.mean(cross_val_score(clf, X_selected[:, curr_features], y, cv=5))

    SU_sel = SU[curr_features]
    ranks = np.argsort(SU_sel)
    features_to_try = curr_features[ranks]

    keep_mask = np.ones(len(curr_features), dtype=bool)

    for i, feat_idx in enumerate(features_to_try):
        temp_mask = keep_mask.copy()
        temp_mask[np.where(curr_features == feat_idx)[0][0]] = 0
        selected_temp = curr_features[temp_mask]

        clf = KNeighborsClassifier(n_neighbors=5)
        acc_temp = np.mean(cross_val_score(clf, X_selected[:, selected_temp], y, cv=5))

        if acc_temp >= base_acc - tol:
            keep_mask[np.where(curr_features == feat_idx)[0][0]] = 0
            base_acc = acc_temp

    curr_features = curr_features[keep_mask]

    improved = True
    while improved:
        improved = False
        for feat_idx in curr_features.copy():
            temp_features = np.setdiff1d(curr_features, [feat_idx])
            clf = KNeighborsClassifier(n_neighbors=5)
            acc_temp = np.mean(cross_val_score(clf, X_selected[:, temp_features], y, cv=5))

            if acc_temp >= base_acc - tol:
                curr_features = temp_features
                base_acc = acc_temp
                improved = True
                break

    final_mask = np.zeros(X.shape[1], dtype=int)
    final_mask[selected_indices[curr_features]] = 1
    return final_mask

# --------- MAIN RUNNER ----------
def run(X, y):
    start = time.time()

    thdoa_best, selected_indices, SU = THDOA(X, y)
    X_reduced = X[:, selected_indices]
    gwo_best = GWO(X_reduced, y, initial_best=thdoa_best)

    pruned_best = final_pruning(X, y, selected_indices, gwo_best, SU)

    final_fitness = fitness(pruned_best, X, y)
    acc = 1 - final_fitness
    feature_count = np.sum(pruned_best)
    elapsed = time.time() - start

    return pruned_best, acc, feature_count, elapsed

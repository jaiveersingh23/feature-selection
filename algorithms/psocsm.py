import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import mutual_info_classif
import time

def fitness(solution, X, y, beta=0.9):
    if np.sum(solution) == 0:
        return 1.0  # Worst possible fitness
    X_sel = X[:, solution == 1]
    clf = KNeighborsClassifier(n_neighbors=5)
    error = 1 - np.mean(cross_val_score(clf, X_sel, y, cv=5))
    feature_ratio = np.sum(solution) / len(solution)
    return beta * error + (1 - beta) * feature_ratio

def binarize(positions, threshold=0.6):
    return (positions > threshold).astype(int)

def piecewise_init(X, y, n_particles):
    SU = mutual_info_classif(X, y)
    ranks = np.argsort(-SU)
    D = X.shape[1]
    M = int(0.1 * D)
    pop = np.zeros((n_particles, D))
    for i in range(n_particles):
        L = max(round((D / n_particles) * (i + 1)), M)
        for d in range(D):
            if d < M:
                pop[i, ranks[d]] = 0.4 * np.random.rand() + 0.6
            elif d < L:
                pop[i, ranks[d]] = np.random.rand()
    return pop, SU, ranks

def comp_score(SU, freq, lam=0.5):
    freq = freq / np.max(freq) if np.max(freq) > 0 else freq
    SU = SU / np.max(SU) if np.max(SU) > 0 else SU
    return lam * freq + (1 - lam) * SU

def reduce_features(SU, freq, gbest_bin, alpha):
    scores = comp_score(SU, freq)
    D = len(SU)
    top_k = int(alpha * D)
    top_indices = np.argsort(-scores)[:top_k]
    selected = np.where(gbest_bin == 1)[0]
    reduced = np.union1d(top_indices, selected)
    return np.array(sorted(reduced))

def update_velocity_position(X, V, P_best, G_best, w, c1, c2):
    r1, r2 = np.random.rand(*X.shape), np.random.rand(*X.shape)
    V_new = w * V + c1 * r1 * (P_best - X) + c2 * r2 * (G_best - X)
    X_new = X + V_new
    return np.clip(X_new, 0, 1), V_new

# ✅ Main PSO-CSM function named run
def run(X_input, y, n_particles=20, max_iter=50):
    start_time = time.time()

    X_full = X_input.copy()  # Full feature matrix for final fitness
    D = X_full.shape[1]

    X = X_full.copy()
    X_pop, SU, _ = piecewise_init(X, y, n_particles)
    V_pop = np.zeros_like(X_pop)
    P_best = X_pop.copy()
    fitness_P = np.array([fitness(binarize(p), X, y) for p in P_best])
    G_idx = np.argmin(fitness_P)
    G_best = P_best[G_idx].copy()
    G_bin = binarize(G_best)
    reduced_indices = np.arange(D)

    for t in range(max_iter):
        w = 0.9 - 0.5 * (t / max_iter)
        X_pop, V_pop = update_velocity_position(X_pop, V_pop, P_best, G_best, w, 1.49445, 1.49445)
        bin_pop = binarize(X_pop)

        fitness_vals = np.array([fitness(b, X, y) for b in bin_pop])
        for i in range(n_particles):
            if fitness_vals[i] < fitness_P[i]:
                P_best[i] = X_pop[i]
                fitness_P[i] = fitness_vals[i]

        G_idx = np.argmin(fitness_P)
        if fitness_P[G_idx] < fitness(G_bin, X, y):
            G_best = X_pop[G_idx]
            G_bin = binarize(G_best)

        # Feature space reduction
        if (t + 1) % (max_iter // 5) == 0:
            freq = np.sum(bin_pop, axis=0)
            if D <= 1000:
                alpha = 0.5 + (1 - 0.5) * np.random.rand()
            elif D <= 5000:
                alpha = 0.1 + (0.5 - 0.1) * np.random.rand()
            else:
                alpha = 0.05 + (0.1 - 0.05) * np.random.rand()
            reduced = reduce_features(SU, freq, G_bin, alpha)

            # Update all components
            X = X[:, reduced]
            SU = SU[reduced]
            X_pop = X_pop[:, reduced]
            V_pop = V_pop[:, reduced]
            P_best = P_best[:, reduced]
            G_best = G_best[reduced]
            G_bin = binarize(G_best)
            reduced_indices = reduced_indices[reduced]

    final_bin = binarize(G_best)
    full_selection = np.zeros(D, dtype=int)
    full_selection[reduced_indices] = final_bin

    # ✅ Evaluate on full dataset with full_selection
    acc = 1 - fitness(full_selection, X_full, y)

    elapsed_time = time.time() - start_time
    return full_selection, acc, np.sum(full_selection), elapsed_time

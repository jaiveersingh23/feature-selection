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

# --------- COMPREHENSIVE SCORING + REDUCTION ----------
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

# --------- GWO ----------
def GWO(X, y, n_agents=30, max_iter=50):
    dim = X.shape[1]
    wolves, SU = piecewise_init(X, y, n_agents)

    fitnesses = np.array([fitness(binarize(wolf), X, y) for wolf in wolves])
    sorted_idx = np.argsort(fitnesses)
    alpha_wolf = wolves[sorted_idx[0]].copy()
    beta_wolf = wolves[sorted_idx[1]].copy()
    delta_wolf = wolves[sorted_idx[2]].copy()
    alpha_fit = fitnesses[sorted_idx[0]]

    for t in range(max_iter):
        a = 2 - t * (2 / max_iter)

        for i in range(n_agents):
            for d in range(dim):
                r1, r2 = np.random.rand(), np.random.rand()

                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * alpha_wolf[d] - wolves[i,d])
                X1 = alpha_wolf[d] - A1 * D_alpha

                r1, r2 = np.random.rand(), np.random.rand()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * beta_wolf[d] - wolves[i,d])
                X2 = beta_wolf[d] - A2 * D_beta

                r1, r2 = np.random.rand(), np.random.rand()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * delta_wolf[d] - wolves[i,d])
                X3 = delta_wolf[d] - A3 * D_delta

                wolves[i,d] = (X1 + X2 + X3) / 3.0

        wolves = np.clip(wolves, 0, 1)

        fitnesses = np.array([fitness(binarize(wolf), X, y) for wolf in wolves])
        sorted_idx = np.argsort(fitnesses)
        if fitnesses[sorted_idx[0]] < alpha_fit:
            alpha_fit = fitnesses[sorted_idx[0]]
            alpha_wolf = wolves[sorted_idx[0]].copy()
            beta_wolf = wolves[sorted_idx[1]].copy()
            delta_wolf = wolves[sorted_idx[2]].copy()

    best_solution = binarize(alpha_wolf)
    return best_solution, SU

# --------- HOA ----------
def HOA(X, y, initial_best, n_agents=20, max_iter=50):
    dim = X.shape[1]
    horses = np.random.randint(0, 2, (n_agents, dim))
    horses[0] = initial_best  # Inject best from GWO

    fitnesses = np.array([fitness(ind, X, y) for ind in horses])
    best = horses[np.argmin(fitnesses)].copy()

    for _ in range(max_iter):
        for i in range(n_agents):
            rand_horse = horses[np.random.randint(n_agents)]
            alpha = np.random.uniform(0.1, 0.5)
            new_horse = (horses[i] + alpha * (best - rand_horse)).astype(int)
            new_horse = np.clip(new_horse, 0, 1)
            new_fit = fitness(new_horse, X, y)
            if new_fit < fitnesses[i]:
                horses[i] = new_horse
                fitnesses[i] = new_fit
                if new_fit < fitness(best, X, y):
                    best = new_horse.copy()
    return best

# --------- MAIN RUNNER for GWO → HOA ----------
def run(X, y):
    start = time.time()

    # Phase 1: GWO
    gwo_best, SU = GWO(X, y)
    freq_dummy = np.ones(len(SU))  # Optional reduction phase (keep your format)
    reduced_indices = np.arange(X.shape[1])  # No reduction — full space

    # Phase 2: HOA
    hoa_best = HOA(X, y, initial_best=gwo_best)

    final_fitness = fitness(hoa_best, X, y)
    acc = 1 - final_fitness
    feature_count = np.sum(hoa_best)
    elapsed = time.time() - start

    return hoa_best, acc, feature_count, elapsed

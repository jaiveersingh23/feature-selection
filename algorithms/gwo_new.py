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
    feature_ranks = np.argsort(-SU)  # descending order
    D = X.shape[1]
    M = max(1, int(0.1 * D))  # at least 1 feature
    agents = np.zeros((n_agents, D))
    for i in range(n_agents):
        L = max(round((D / n_agents) * (i + 1)), M)
        for d in range(D):
            if d < M:
                agents[i, feature_ranks[d]] = 0.4 * np.random.rand() + 0.6  # [0.6,1.0]
            elif d < L:
                agents[i, feature_ranks[d]] = np.random.rand()  # [0,1]
    return np.clip(agents, 0, 1), SU

# --------- GREY WOLF OPTIMIZATION (with piecewise init) ----------
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
        a = 2 - t * (2 / max_iter)  # linearly decreases from 2 to 0

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
    return best_solution

# --------- MAIN RUNNER ----------
def run(X, y):
    start = time.time()
    best_solution = GWO(X, y)
    final_fitness = fitness(best_solution, X, y)
    acc = 1 - final_fitness  # Accuracy approx
    feature_count = np.sum(best_solution)
    elapsed = time.time() - start
    return best_solution, acc, feature_count, elapsed

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import time

def fitness(solution, X, y, alpha=0.98):
    if np.sum(solution) == 0:
        return 0
    acc = np.mean(cross_val_score(KNeighborsClassifier(n_neighbors=5), X[:, solution == 1], y, cv=5))
    feature_ratio = np.sum(solution) / len(solution)
    return alpha * acc + (1 - alpha) * (1 - feature_ratio)

def binarize(position, threshold=0.5):
    return (position > threshold).astype(int)

def GWO(X, y, n_agents=30, max_iter=50):
    dim = X.shape[1]

    # Initialize positions of wolves randomly in [0,1]
    wolves = np.random.rand(n_agents, dim)

    # Initialize alpha, beta, delta wolves
    fitnesses = np.array([fitness(binarize(wolf), X, y) for wolf in wolves])
    sorted_idx = np.argsort(fitnesses)[::-1]
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

        sorted_idx = np.argsort(fitnesses)[::-1]
        if fitnesses[sorted_idx[0]] > alpha_fit:
            alpha_fit = fitnesses[sorted_idx[0]]
            alpha_wolf = wolves[sorted_idx[0]].copy()
            beta_wolf = wolves[sorted_idx[1]].copy()
            delta_wolf = wolves[sorted_idx[2]].copy()

    best_solution = binarize(alpha_wolf)
    return best_solution

def run(X, y):
    start = time.time()
    best_solution = GWO(X, y)
    acc = fitness(best_solution, X, y)
    feature_count = np.sum(best_solution)
    elapsed = time.time() - start
    return best_solution, acc, feature_count, elapsed

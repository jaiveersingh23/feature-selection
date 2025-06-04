import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import time

# ---------- Fitness with Feature Count Penalty ----------
def fitness(solution, X, y, alpha=0.98):
    if np.sum(solution) == 0:
        return 0
    X_selected = X[:, solution == 1]
    clf = KNeighborsClassifier(n_neighbors=5)
    acc = np.mean(cross_val_score(clf, X_selected, y, cv=5))
    feature_ratio = np.sum(solution) / X.shape[1]
    return alpha * acc + (1 - alpha) * (1 - feature_ratio)

# ---------- Binarization with Adaptive Threshold ----------
def binarize(position, threshold=0.5):
    return (position > threshold).astype(int)

# ---------- THDOA Functions ----------
def heat_diffusion(agents, diffusion_rate=0.1):
    new_agents = agents.copy()
    for i in range(len(agents)):
        neighbors = np.random.choice(len(agents), size=3, replace=False)
        neighbor_avg = np.mean(agents[neighbors], axis=0)
        new_agents[i] += diffusion_rate * (neighbor_avg - agents[i])
    return np.clip(new_agents, 0, 1)

def cooling(agents, cooling_factor=0.99):
    return agents * cooling_factor

# ---------- GWO ----------
def gwo_update(agents, alpha, beta, delta):
    a = 2
    new_agents = np.zeros_like(agents)
    for i in range(len(agents)):
        r1, r2 = np.random.rand(), np.random.rand()
        A1 = 2 * a * r1 - a
        C1 = 2 * r2
        D_alpha = abs(C1 * alpha - agents[i])
        X1 = alpha - A1 * D_alpha

        r1, r2 = np.random.rand(), np.random.rand()
        A2 = 2 * a * r1 - a
        C2 = 2 * r2
        D_beta = abs(C2 * beta - agents[i])
        X2 = beta - A2 * D_beta

        r1, r2 = np.random.rand(), np.random.rand()
        A3 = 2 * a * r1 - a
        C3 = 2 * r2
        D_delta = abs(C3 * delta - agents[i])
        X3 = delta - A3 * D_delta

        new_agents[i] = (X1 + X2 + X3) / 3
    return np.clip(new_agents, 0, 1)

# ---------- HOA ----------
def hoa_update(horses, best, X, y, alpha_range=(0.1, 0.5)):
    new_horses = horses.copy()
    for i in range(len(horses)):
        rand_horse = horses[np.random.randint(len(horses))]
        alpha = np.random.uniform(*alpha_range)
        new_horse = (horses[i] + alpha * (best - rand_horse)).astype(int)
        new_horse = np.clip(new_horse, 0, 1)
        if fitness(new_horse, X, y) > fitness(horses[i], X, y):
            new_horses[i] = new_horse
    return new_horses

# ---------- Hybrid THDOA + GWO + HOA ----------
def hybrid_THDOA_GWO_HOA(X, y, n_agents=20, max_iter=60, alpha=0.98):
    dim = X.shape[1]
    agents = np.random.rand(n_agents, dim)
    elite_agent = None
    elite_score = -np.inf
    elite_features = dim

    for iter in range(max_iter):
        threshold = 0.6 - 0.3 * (iter / max_iter)  # Adaptive binarization

        if iter < max_iter // 3:
            # THDOA Phase
            agents = heat_diffusion(agents)
            agents = cooling(agents)
        elif iter < 2 * max_iter // 3:
            # GWO Phase
            bin_agents = np.array([binarize(a, threshold) for a in agents])
            fit_values = np.array([fitness(a, X, y, alpha) for a in bin_agents])
            sorted_idx = np.argsort(fit_values)[::-1]
            alpha_wolf = agents[sorted_idx[0]]
            beta_wolf = agents[sorted_idx[1]]
            delta_wolf = agents[sorted_idx[2]]
            agents = gwo_update(agents, alpha_wolf, beta_wolf, delta_wolf)
        else:
            # HOA Phase
            bin_agents = np.array([binarize(a, threshold) for a in agents])
            fit_values = np.array([fitness(a, X, y, alpha) for a in bin_agents])
            best_idx = np.argmax(fit_values)
            best_horse = bin_agents[best_idx]
            agents = hoa_update(bin_agents, best_horse, X, y)

        # Track elite agent
        bin_agents = np.array([binarize(a, threshold) for a in agents])
        fit_values = np.array([fitness(a, X, y, alpha) for a in bin_agents])
        for i, f in enumerate(fit_values):
            feat_count = np.sum(bin_agents[i])
            if f > elite_score or (f == elite_score and feat_count < elite_features):
                elite_agent = agents[i].copy()
                elite_score = f
                elite_features = feat_count

    return elite_agent

# ---------- Final Runner ----------
def run(X, y, alpha=0.98):
    start = time.time()
    best = hybrid_THDOA_GWO_HOA(X, y, alpha=alpha)
    final_threshold = 0.5  # Apply final binarization threshold
    bin_best = binarize(best, final_threshold)
    acc = fitness(bin_best, X, y, alpha=alpha)
    return bin_best, acc, np.sum(bin_best), time.time() - start

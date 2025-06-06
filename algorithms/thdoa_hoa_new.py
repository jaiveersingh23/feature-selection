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

# --------- THDOA ----------
def THDOA(X, y, n_agents=20, max_iter=50):
    dim = X.shape[1]
    agents, SU = piecewise_init(X, y, n_agents)
    reduced_indices = np.arange(dim)  # Tracks active features

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
            best_bin = binarize(best_agent)

    return binarize(best_agent), reduced_indices

# --------- HOA ----------
def HOA(X, y, initial_best, n_agents=20, max_iter=50):
    dim = X.shape[1]
    horses = np.random.randint(0, 2, (n_agents, dim))
    horses[0] = initial_best  # Inject best from THDOA

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

# --------- MAIN RUNNER ----------
def run(X, y):
    start = time.time()

    # Phase 1: THDOA
    thdoa_best, selected_indices = THDOA(X, y)

    # Restore full length vector
    full_best = np.zeros(X.shape[1], dtype=int)
    full_best[selected_indices] = thdoa_best

    # Phase 2: HOA
    hybrid_best = HOA(X, y, initial_best=full_best)

    final_fitness = fitness(hybrid_best, X, y)
    acc = 1 - final_fitness  # Approximate accuracy
    feature_count = np.sum(hybrid_best)
    elapsed = time.time() - start

    return hybrid_best, acc, feature_count, elapsed

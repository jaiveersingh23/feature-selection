import numpy as np
import time
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

# ----- Fitness Function -----
def fitness(solution, X, y):
    if np.sum(solution) == 0:
        return 0
    X_selected = X[:, solution == 1]
    clf = KNeighborsClassifier(n_neighbors=5)
    return np.mean(cross_val_score(clf, X_selected, y, cv=5))

# ----- Binarization -----
def binarize(position):
    return (position > 0.5).astype(int)

# ----- THDOA Core -----
def heat_diffusion(agents, fitness_values, diffusion_rate=0.1):
    new_agents = agents.copy()
    for i in range(len(agents)):
        neighbors = np.random.choice(len(agents), size=3, replace=False)
        neighbor_avg = np.mean(agents[neighbors], axis=0)
        new_agents[i] += diffusion_rate * (neighbor_avg - agents[i])
    return np.clip(new_agents, 0, 1)

def cooling(agents, cooling_factor=0.99):
    return agents * cooling_factor

def THDOA(X, y, n_agents=20, max_iter=50):
    dim = X.shape[1]
    agents = np.random.rand(n_agents, dim)
    fitness_values = np.array([fitness(binarize(agent), X, y) for agent in agents])

    best_idx = np.argmax(fitness_values)
    best_agent = agents[best_idx].copy()
    best_fitness = fitness_values[best_idx]

    for _ in range(max_iter):
        agents = heat_diffusion(agents, fitness_values)
        agents = cooling(agents)
        fitness_values = np.array([fitness(binarize(agent), X, y) for agent in agents])
        current_best_idx = np.argmax(fitness_values)
        if fitness_values[current_best_idx] > best_fitness:
            best_fitness = fitness_values[current_best_idx]
            best_agent = agents[current_best_idx].copy()

    return binarize(best_agent)

# ----- SSO Core -----
def SSO(X, y, initial_best, n_agents=20, max_iter=50):
    dim = X.shape[1]
    population = np.random.randint(0, 2, (n_agents, dim))
    population[0] = initial_best.copy()

    fitnesses = np.array([fitness(ind, X, y) for ind in population])
    best_idx = np.argmax(fitnesses)
    best = population[best_idx].copy()

    for _ in range(max_iter):
        for i in range(n_agents):
            sperm = population[i]
            mutation = np.random.rand(dim) < 0.1
            new_sperm = sperm ^ mutation.astype(int)
            new_fit = fitness(new_sperm, X, y)
            if new_fit > fitnesses[i]:
                population[i] = new_sperm
                fitnesses[i] = new_fit
                if new_fit > fitness(best, X, y):
                    best = new_sperm.copy()
    return best

# ----- Hybrid Run -----
def run(X, y, thdoa_iters=50, sso_iters=50):
    start = time.time()

    # Phase 1: THDOA exploration
    thdoa_best = THDOA(X, y, max_iter=thdoa_iters)

    # Phase 2: SSO refinement
    hybrid_best = SSO(X, y, initial_best=thdoa_best, max_iter=sso_iters)

    acc = fitness(hybrid_best, X, y)
    feature_count = np.sum(hybrid_best)
    elapsed = time.time() - start

    return hybrid_best, acc, feature_count, elapsed

import numpy as np
import time
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import mutual_info_classif

# ----- SU-based Initialization -----
def su_based_initialization(X, y, n_agents=20, top_percent=0.1, guided_ratio=0.3):
    dim = X.shape[1]
    agents = np.random.randint(0, 2, (n_agents, dim))

    su_scores = mutual_info_classif(X, y, discrete_features='auto')
    top_k = int(top_percent * dim)
    top_indices = np.argsort(su_scores)[-top_k:]

    n_guided = int(guided_ratio * n_agents)
    for i in range(n_guided):
        agents[i] = np.zeros(dim)
        agents[i][top_indices] = 1

    return agents.astype(float)  # for THDOA heat diffusion

# ----- Fitness Function with CSM memory support -----
def fitness(solution, X, y, beta=0.9, memory=None):
    if np.sum(solution) == 0:
        return 0
    X_selected = X[:, solution == 1]
    clf = KNeighborsClassifier(n_neighbors=1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc = np.mean(cross_val_score(clf, X_selected, y, cv=cv))
    error = 1 - acc
    feat_ratio = np.sum(solution) / len(solution)
    
    if memory is not None:
        memory += solution  # accumulate frequency of feature selection

    cost = beta * error + (1 - beta) * feat_ratio
    return 1 - cost  # maximize

# ----- Binarize Agent -----
def binarize(position):
    return (position > 0.5).astype(int)

# ----- Heat Diffusion -----
def heat_diffusion(agents, diffusion_rate=0.1):
    new_agents = agents.copy()
    for i in range(len(agents)):
        neighbors = np.random.choice(len(agents), size=3, replace=False)
        neighbor_avg = np.mean(agents[neighbors], axis=0)
        new_agents[i] += diffusion_rate * (neighbor_avg - agents[i])
    return np.clip(new_agents, 0, 1)

# ----- Cooling Step -----
def cooling(agents, cooling_factor=0.99):
    return agents * cooling_factor

# ----- THDOA with SU Init + CSM -----
def THDOA(X, y, n_agents=20, max_iter=50, beta=0.9):
    dim = X.shape[1]
    agents = su_based_initialization(X, y, n_agents=n_agents)
    memory = np.zeros(dim)
    fitness_values = np.array([fitness(binarize(agent), X, y, beta, memory) for agent in agents])
    
    best_idx = np.argmax(fitness_values)
    best_agent = agents[best_idx].copy()
    best_fitness = fitness_values[best_idx]

    for _ in range(max_iter):
        agents = heat_diffusion(agents)
        agents = cooling(agents)
        fitness_values = np.array([fitness(binarize(agent), X, y, beta, memory) for agent in agents])
        current_best_idx = np.argmax(fitness_values)
        if fitness_values[current_best_idx] > best_fitness:
            best_fitness = fitness_values[current_best_idx]
            best_agent = agents[current_best_idx].copy()

    memory = memory / (n_agents * max_iter)  # normalize CSM memory
    return binarize(best_agent), memory

# ----- HOA with CSM Influence -----
def HOA(X, y, initial_best, memory, n_agents=20, max_iter=50, beta=0.9):
    dim = X.shape[1]
    horses = np.random.randint(0, 2, (n_agents, dim))
    horses[0] = initial_best

    fitnesses = np.array([fitness(ind, X, y, beta) for ind in horses])
    best = horses[np.argmax(fitnesses)].copy()

    for _ in range(max_iter):
        for i in range(n_agents):
            rand_horse = horses[np.random.randint(n_agents)]
            alpha = np.random.uniform(0.1, 0.5)
            influence = memory > np.mean(memory)  # use CSM to guide toward commonly selected features
            new_horse = (horses[i] + alpha * (best - rand_horse)).astype(int)
            new_horse[influence] = 1  # promote frequently selected features
            new_horse = np.clip(new_horse, 0, 1)
            new_fit = fitness(new_horse, X, y, beta)
            if new_fit > fitnesses[i]:
                horses[i] = new_horse
                fitnesses[i] = new_fit
                if new_fit > fitness(best, X, y, beta):
                    best = new_horse.copy()
    return best

# ----- Full Hybrid Run -----
def run(X, y, beta=0.95):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7,
                                                        stratify=y, random_state=42)

    start = time.time()
    thdoa_best, memory = THDOA(X_train, y_train, beta=beta)
    hybrid_best = HOA(X_train, y_train, initial_best=thdoa_best, memory=memory, beta=beta)
    elapsed = time.time() - start

    selected_features = np.sum(hybrid_best)
    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(X_train[:, hybrid_best == 1], y_train)
    acc = clf.score(X_test[:, hybrid_best == 1], y_test)

    return hybrid_best, acc, selected_features, elapsed

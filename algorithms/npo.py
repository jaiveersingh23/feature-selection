import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import time

# ---------- FITNESS FUNCTION ----------
def fitness(solution, X, y, alpha=0.98):
    if np.sum(solution) == 0:
        return 0
    acc = np.mean(cross_val_score(KNeighborsClassifier(n_neighbors=5), X[:, solution == 1], y, cv=5))
    feature_ratio = np.sum(solution) / len(solution)
    return alpha * acc + (1 - alpha) * (1 - feature_ratio)

# ---------- BINARIZATION ----------
def binarize(position):
    return (position > 0.5).astype(int)

# ---------- NPO COMPONENTS ----------
def update_membrane_potential(V, I, L, alpha=0.9, beta=0.5, gamma=0.3, delta=0.1):
    noise = np.random.normal(0, 1, size=V.shape)
    V_new = alpha * V + beta * I + gamma * L + delta * noise
    return V_new

def lateral_inhibition(S, W):
    # Compute L_i term (sum of W_ij * s_j)
    return np.dot(W, S)

# ---------- MAIN NPO OPTIMIZER ----------
def NPO(X, y, n_agents=20, max_iter=50):
    dim = X.shape[1]
    
    # Initialize agents (membrane potential V), firing threshold theta, weights W
    V = np.random.rand(n_agents, dim)
    theta = np.ones((n_agents, dim)) * 1.0
    agents = np.zeros((n_agents, dim))  # binary spike output
    
    # Initialize W_ij based on random correlation (optional: you can use MI if needed)
    W = np.random.uniform(-0.2, 0.2, size=(dim, dim))
    np.fill_diagonal(W, 0)  # no self-loop
    
    # Initial firing decision
    agents = binarize(V)
    
    # Evaluate initial fitness
    fitness_values = np.array([fitness(agent, X, y) for agent in agents])
    best_idx = np.argmax(fitness_values)
    best_agent = agents[best_idx].copy()
    best_fitness = fitness_values[best_idx]
    
    for _ in range(max_iter):
        for i in range(n_agents):
            # Compute input I_i = feature relevance (simple version: 1 - correlation with target)
            I = np.random.uniform(0.5, 1.0, size=dim)  # You can replace with MI-based input
            
            # Lateral inhibition/excitation term
            L = lateral_inhibition(agents[i], W)
            
            # Update membrane potential
            V[i] = update_membrane_potential(V[i], I, L)
            
            # Firing decision
            agents[i] = (V[i] >= theta[i]).astype(int)
            
            # Optional: adapt threshold slightly (to avoid too frequent firing)
            theta[i] += 0.05 * (agents[i] - 0.5)
        
        # Evaluate fitness
        fitness_values = np.array([fitness(agent, X, y) for agent in agents])
        current_best_idx = np.argmax(fitness_values)
        
        if fitness_values[current_best_idx] > best_fitness:
            best_fitness = fitness_values[current_best_idx]
            best_agent = agents[current_best_idx].copy()
    
    return best_agent

# ---------- RUN FUNCTION ----------
def run(X, y):
    start = time.time()
    
    # Phase 1: Run NPO to select best features
    npo_best = NPO(X, y)
    
    acc = fitness(npo_best, X, y)
    feature_count = np.sum(npo_best)
    elapsed = time.time() - start
    
    return npo_best, acc, feature_count, elapsed

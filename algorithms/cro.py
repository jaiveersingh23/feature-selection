import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import time

def fitness(solution, X, y):
    """Evaluate fitness (accuracy) of a binary feature subset using KNN."""
    selected_features = np.where(solution == 1)[0]  # Indices of selected features
    if len(selected_features) == 0:
        return 0  # No features selected, return a low fitness value
    X_selected = X[:, selected_features]
    clf = KNeighborsClassifier(n_neighbors=5)
    score = np.mean(cross_val_score(clf, X_selected, y, cv=5))
    return score

def resonance(solution, f_s, f_optimal, f_max, f_min):
    """Calculate the resonance factor based on the fitness of the solution."""
    if f_max == f_min:
        return 0  # Avoid division by zero
    return 1 - (abs(f_s - f_optimal) / (f_max - f_min))

def dissonance(solution, f_s, f_optimal, f_max, f_min, delta=0.5):
    """Calculate dissonance adjustment based on the fitness of the solution."""
    if f_max == f_min:
        return 0  # Avoid division by zero
    return (abs(f_s - f_optimal) / (f_max - f_min)) * delta

def cognitive_update(solution, R, D, grad_f, grad_g, alpha=0.1):
    """Update the solution based on resonance and dissonance."""
    updated_solution = solution + alpha * (R * grad_f - D * grad_g)
    # Ensure the solution is binary (0 or 1) and within the feature bounds
    updated_solution = np.clip(updated_solution, 0, 1)
    return updated_solution

def CRO(X, y, n_agents=20, max_iter=50):
    """Cognitive Resonance Optimization algorithm."""
    dim = X.shape[1]
    population = np.random.rand(n_agents, dim)  # Initialize population with random values
    population = np.round(population)  # Convert to binary (0 or 1)
    fitnesses = np.array([fitness(ind, X, y) for ind in population])

    # Initialize the best solution
    best_idx = np.argmax(fitnesses)
    best = population[best_idx].copy()

    # Optimal fitness value
    f_optimal = np.max(fitnesses)

    for _ in range(max_iter):
        for i in range(n_agents):
            f_s = fitnesses[i]
            R = resonance(population[i], f_s, f_optimal, np.max(fitnesses), np.min(fitnesses))
            D = dissonance(population[i], f_s, f_optimal, np.max(fitnesses), np.min(fitnesses))

            # Set gradients to 1 (simplified approach for now)
            grad_f = 1
            grad_g = 1

            # Update the agent using cognitive influence
            new_solution = cognitive_update(population[i], R, D, grad_f, grad_g)

            # Ensure at least one feature is selected
            if np.sum(new_solution) == 0:
                # If no features are selected, force the selection of at least one feature
                new_solution[np.random.randint(0, dim)] = 1

            population[i] = new_solution

        # Recalculate fitness values after updating agents
        fitnesses = np.array([fitness(ind, X, y) for ind in population])

        # Update the best solution and optimal fitness
        f_optimal = np.max(fitnesses)
        best = population[np.argmax(fitnesses)].copy()

    return best

def run(X, y):
    """Run the Cognitive Resonance Optimization algorithm."""
    start = time.time()
    best = CRO(X, y)
    acc = fitness(best, X, y)
    return best, acc, np.sum(best), time.time() - start

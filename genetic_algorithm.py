import numpy as np
from copy import deepcopy

from data_utils import initialize_binary_vector, select_features
from fitness import compute_fitness

#------------------------ Selection -----------------------#

def tournament_selection(population, fitness_scores, tournament_size=3):

    """
    Perform tournament selection to choose an individual from the population.
    Args:
        population (ndarray): Current population of binary vectors.
        fitness_scores (ndarray): Corresponding fitness scores for the population.
        tournament_size (int): Number of individuals to compete in each tournament.
    Returns:
        selected_individual (ndarray)
    """
    indices = np.random.choice(len(population), tournament_size, replace=False)
    best_idx = indices[np.argmax(fitness_scores[indices])]
    return deepcopy(population[best_idx])

#------------------------ Crossover -----------------------#

def one_point_crossover(parent1, parent2):
    """
    Perform one-point crossover between two parents to produce two offspring.
    Args:
        parent1 (ndarray): First parent binary vector.
        parent2 (ndarray): Second parent binary vector.
    Returns:
        child1 (ndarray)
        child2 (ndarray)
    """
    n_genes = len(parent1)
    point = np.random.randint(1, n_genes - 1)
    child1 = np.concatenate([parent1[:point], parent2[point:]])
    child2 = np.concatenate([parent2[:point], parent1[point:]])
    return child1, child2

#------------------------ Mutation -----------------------#

def mutate(individual, mutation_rate):
    """
    Perform bit-flip mutation on an individual.
    Args:
        individual (ndarray): Binary vector to mutate.
        mutation_rate (float): Probability of flipping each bit.
    Returns:
        mutated_individual (ndarray)
    """
    mutant = individual
    for i in range(len(mutant)):
        if np.random.rand() < mutation_rate:
            mutant[i] = 1 - mutant[i]  # Flip bit
    return mutant


#----------------------- GA -----------------------#

def run_genetic_algorithm(
    X,
    y,
    pop_size=20,
    num_generations=30,
    mutation_rate=0.05,
    alpha=0.01,
    k_folds=5,
    mode="exploratory",
    tournament_size=3,
    random_state=None
):
    """
    Run a Genetic Algorithm (GA) for feature selection with two modes:
    
    Modes:
        - "elitist": standard GA with elitism; each generation retains the best individuals
                     to converge faster to high-fitness solutions.
        - "exploratory": GA adapted for scientific feature analysis; all individuals
                         are stored across generations to perform gene ranking.
    
    Args:
        X, y (ndarray): Feature matrix and labels
        pop_size (int): Number of individuals per generation
        num_generations (int)
        mutation_rate (float)
        alpha (float): Penalization coefficient
        k_folds (int): Number of folds for cross-validation
        mode (str): "elitist" or "exploratory"
        tournament_size (int)
        random_state (int)
    
    Returns:
        Depending on mode:
            - "elitist": best_individual (ndarray), best_fitness (float)
            - "exploratory": all_individuals (list of ndarray), all_fitnesses (list of float)
    """
    if random_state is not None:
        np.random.seed(random_state)

    num_features = X.shape[1]

    if mode == "exploratory":
        # Keep all individuals and fitness for gene ranking
        all_individuals = []
        all_fitnesses = []

        for gen in range(num_generations):
            # Generate a new population
            population = initialize_binary_vector(pop_size, num_features, random_state)
            fitness_scores = []

            for ind in population:
                fitness = compute_fitness(ind, X, y, k_folds, alpha, random_state)
                fitness_scores.append(fitness)

            fitness_scores = np.array(fitness_scores)
            all_individuals.extend(population)
            all_fitnesses.extend(fitness_scores)

            print(f"Generation {gen+1}: Best fitness = {fitness_scores.max():.4f}")

        return all_individuals, all_fitnesses

    elif mode == "elitist":
        """
        Standard GA with elitism: keep top individuals each generation.
        This version ensures that multiple relevant features are selected
        by combining top candidates with maximal fitness.
        """
        population = initialize_binary_vector(pop_size, num_features, random_state)
        fitness_scores = np.array([compute_fitness(ind, X, y, k_folds, alpha, random_state)
                                for ind in population])

        for gen in range(num_generations):
            new_population = []

            while len(new_population) < pop_size:
                parent1 = tournament_selection(population, fitness_scores, tournament_size)
                parent2 = tournament_selection(population, fitness_scores, tournament_size)

                child1, child2 = one_point_crossover(parent1, parent2)
                child1 = mutate(child1, mutation_rate)
                child2 = mutate(child2, mutation_rate)

                new_population.extend([child1, child2])

            # Combine parents + children
            combined_population = np.vstack([population, new_population])
            combined_fitness = np.array([compute_fitness(ind, X, y, k_folds, alpha, random_state)
                                        for ind in combined_population])

            # Keep the top pop_size individuals
            best_indices = np.argsort(combined_fitness)[-pop_size:]
            population = combined_population[best_indices]
            fitness_scores = combined_fitness[best_indices]

            print(f"Generation {gen+1}: Best fitness = {fitness_scores.max():.4f}")

        # ------------------------ Combine top candidates to select multiple features ------------------------ #
        max_fitness = fitness_scores.max()
        candidates = population[fitness_scores == max_fitness]

        # Calculate feature frequency in top candidates
        feature_frequency = candidates.sum(axis=0) / len(candidates)

        # Select features that appear in at least 30% of top individuals
        threshold = 0.3
        best_individual = (feature_frequency >= threshold).astype(int)
        best_fitness = max_fitness

        return best_individual, best_fitness

    else:
        raise ValueError("mode must be 'exploratory' or 'elitist'")



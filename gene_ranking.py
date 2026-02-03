import numpy as np
from data_utils import track_indices, get_selected_feature_names
from fitness import compute_fitness

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ----------------------- Gene frequency ranking ----------------------- #
def rank_genes(all_individuals, all_fitnesses):
    """
    Compute frequency of each gene (feature) in the exploratory GA population.

    Args:
        all_individuals (list of ndarray): All individuals from exploratory GA
        all_fitnesses (list of float): Corresponding fitnesses

    Returns:
        ranked_indices (list): Feature indices sorted by frequency (descending)
        gene_frequency (dict): Mapping of index -> frequency
    """
    num_features = len(all_individuals[0])
    counts = np.zeros(num_features)

    # Weight by fitness optionally (here simple count)
    for ind in all_individuals:
        selected = track_indices(ind)
        counts[selected] += 1  # increment frequency

    ranked_indices = np.argsort(-counts)  # descending order
    gene_frequency = {i: counts[i] for i in range(num_features)}

    return ranked_indices, gene_frequency

# ----------------------- Progressive model ----------------------- #
def progressive_feature_selection(X, y, ranked_indices, alpha=0.01, k_folds=5):
    """
    Build models progressively adding features by ranking and check performance.

    Args:
        X, y: Dataset
        ranked_indices (list or ndarray): Features ordered by importance
        alpha (float): Penalization coefficient
        k_folds (int)

    Returns:
        selected_features_indices: minimal set of features maintaining near-max performance
        performance_list: list of (num_features, fitness) tuples
    """
    selected_features_indices = []
    performance_list = []

    best_fitness_so_far = -np.inf

    for idx in ranked_indices:
        selected_features_indices.append(idx)

        # Build a temporary binary vector for fitness calculation
        individual_vector = np.zeros(X.shape[1], dtype=int)
        individual_vector[selected_features_indices] = 1

        # Compute fitness
        fitness = compute_fitness(individual_vector, X, y, k_folds=k_folds, alpha=alpha)

        performance_list.append((len(selected_features_indices), fitness))

        # Update best fitness so far
        if fitness > best_fitness_so_far:
            best_fitness_so_far = fitness
        else:
            # Optional: stop adding if no improvement (cutoff)
            break

    return selected_features_indices, performance_list

# ----------------------- Map indices to feature names ----------------------- #
def get_feature_names_from_indices(feature_names, selected_indices):
    """
    Convert selected feature indices to names.

    Args:
        feature_names (array-like)
        selected_indices (list or array)

    Returns:
        list of feature names
    """
    return get_selected_feature_names(feature_names, np.array(selected_indices))
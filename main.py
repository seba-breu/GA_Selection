import numpy as np
import matplotlib.pyplot as plt

from data_utils import (
    load_breast_cancer,
    train_val_split,
    normalize_features,
    track_indices,
    get_selected_feature_names
)
from genetic_algorithm import run_genetic_algorithm
from fitness import compute_fitness
from gene_ranking import rank_genes, progressive_feature_selection, get_feature_names_from_indices

# ------------------------ User Parameters ------------------------ #
POP_SIZE = 50
NUM_GENERATIONS = 30
MUTATION_RATE = 0.05
ALPHA = 0.01
K_FOLDS = 5
RANDOM_STATE = 42

# Ask user for GA mode
MODE = input("Choose GA mode ('elitist' or 'exploratory'): ").strip().lower()
if MODE not in ["elitist", "exploratory"]:
    raise ValueError("Mode must be 'elitist' or 'exploratory'")

# ------------------------ Load and Prepare Data ------------------------ #
X, y = load_breast_cancer(as_dataframe=True)
X_train, X_val, y_train, y_val = train_val_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=True
)
X_train_norm, X_val_norm = normalize_features(X_train, X_val)

# ------------------------ Run Genetic Algorithm ------------------------ #
if MODE == "elitist":
    # Run elitist GA: returns best individual only
    best_individual, best_fitness = run_genetic_algorithm(
        X_train_norm,
        y_train,
        pop_size=POP_SIZE,
        num_generations=NUM_GENERATIONS,
        mutation_rate=MUTATION_RATE,
        alpha=ALPHA,
        k_folds=K_FOLDS,
        mode="elitist",
        tournament_size=3,
        random_state=RANDOM_STATE
    )

    print(f"\nElitist GA completed. Best fitness: {best_fitness:.4f}")

    # Extract selected features
    selected_indices = track_indices(best_individual)
    X_train_sel = X_train_norm[:, selected_indices]
    X_val_sel = X_val_norm[:, selected_indices]

    # Map indices to feature names
    selected_feature_names = get_selected_feature_names(X_train.columns, selected_indices)
    print("Selected feature names:", selected_feature_names)

else:  # exploratory mode
    # Run exploratory GA: returns all individuals and fitnesses
    all_individuals, all_fitnesses = run_genetic_algorithm(
        X_train_norm,
        y_train,
        pop_size=POP_SIZE,
        num_generations=NUM_GENERATIONS,
        mutation_rate=MUTATION_RATE,
        alpha=ALPHA,
        k_folds=K_FOLDS,
        mode="exploratory",
        tournament_size=3,
        random_state=RANDOM_STATE
    )

    print(f"\nExploratory GA completed: {len(all_individuals)} individuals evaluated")

    # ------------------------ Gene Ranking ------------------------ #
    ranked_indices, gene_frequency = rank_genes(all_individuals, all_fitnesses)
    print("Top 10 ranked feature indices:", ranked_indices[:10])

    # Plot top gene frequencies
    top_genes = ranked_indices[:15]
    top_freqs = [gene_frequency[i] for i in top_genes]

    plt.figure(figsize=(10,5))
    plt.bar(range(len(top_genes)), top_freqs, tick_label=top_genes)
    plt.xlabel("Feature Index")
    plt.ylabel("Frequency in GA Population")
    plt.title("Top 15 Feature Frequencies in Exploratory GA")
    plt.savefig("gene_frequencies.png", dpi=300, bbox_inches='tight')  # save figure
    plt.show()

    # ------------------------ Progressive Feature Selection ------------------------ #
    selected_indices, performance_list = progressive_feature_selection(
        X_train_norm,
        y_train,
        ranked_indices,
        alpha=ALPHA,
        k_folds=K_FOLDS
    )

    selected_feature_names = get_feature_names_from_indices(X_train.columns, selected_indices)
    print("Selected feature names after progressive selection:", selected_feature_names)
    print("Performance at each step (num_features, fitness):", performance_list)

    # Plot performance progression
    num_features_list = [x[0] for x in performance_list]
    fitness_list = [x[1] for x in performance_list]

    plt.figure(figsize=(8,4))
    plt.plot(num_features_list, fitness_list, marker='o')
    plt.xlabel("Number of Features Selected")
    plt.ylabel("Fitness")
    plt.title("Progressive Feature Selection Performance")
    plt.grid(True)
    plt.savefig("progressive_selection.png", dpi=300, bbox_inches='tight')  # save figure
    plt.show()

    # ------------------------ Prepare Final Datasets ------------------------ #
    X_train_sel = X_train_norm[:, selected_indices]
    X_val_sel = X_val_norm[:, selected_indices]

# ------------------------ Final Dataset Info ------------------------ #
print("\nFinal X_train shape:", X_train_sel.shape)
print("Final X_val shape:", X_val_sel.shape)

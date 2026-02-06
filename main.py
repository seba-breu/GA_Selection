import numpy as np
import matplotlib.pyplot as plt

from data_utils import (
    load_breast_cancer_data,
    train_val_split,
    normalize_features,
    track_indices,
    get_selected_feature_names
)

from genetic_algorithm import run_genetic_algorithm
from gene_ranking import (
    rank_genes,
    progressive_feature_selection
)

# ------------------------ GA Mode ------------------------ #
MODE = input("Choose GA mode ('elitist' or 'exploratory'): ").strip().lower()
if MODE not in ["elitist", "exploratory"]:
    raise ValueError("Mode must be 'elitist' or 'exploratory'")

# ------------------------ Load and Prepare Data ------------------------ #
X, y, feature_names = load_breast_cancer_data(as_dataframe=True)

X = X.values
y = y.values

X_train, X_val, y_train, y_val = train_val_split(
    X,
    y,
    test_size=0.2,
    stratify=True
)

X_train_norm, X_val_norm = normalize_features(X_train, X_val)

# ------------------------ Run Genetic Algorithm ------------------------ #
if MODE == "elitist":

    best_individual, best_fitness = run_genetic_algorithm(
        X_train_norm,
        y_train,
        mode="elitist"
    )

    print(f"\nElitist GA completed. Best fitness: {best_fitness:.4f}")

    selected_indices = track_indices(best_individual)

    X_train_sel = X_train_norm[:, selected_indices]
    X_val_sel = X_val_norm[:, selected_indices]

    selected_feature_names = get_selected_feature_names(
        feature_names,
        best_individual
    )

    print("Selected features:")
    for f in selected_feature_names:
        print("-", f)

else:
    all_individuals, all_fitnesses = run_genetic_algorithm(
        X_train_norm,
        y_train,
        mode="exploratory"
    )

    print(f"\nExploratory GA completed: {len(all_individuals)} individuals evaluated")

    # ------------------------ Gene Ranking ------------------------ #
    ranked_indices, gene_frequency = rank_genes(
        all_individuals,
        all_fitnesses
    )

    top_10_indices = ranked_indices[:10]
    top_10_names = [str(feature_names[i]) for i in top_10_indices]

    print("Top 10 ranked feature names:")
    for f in top_10_names:
        print(" -", f)

    # ------------------------ Plot Gene Frequencies ------------------------ #
    top_10_freqs = [gene_frequency[i] for i in top_10_indices]

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(top_10_names)), top_10_freqs, tick_label=top_10_names)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Feature name")
    plt.ylabel("Frequency in GA population")
    plt.title("Top 10 Features selected during exploratory GA")
    plt.tight_layout()
    plt.savefig("gene_frequencies.png", dpi=300)
    plt.show()

    # ------------------------ Progressive Feature Selection ------------------------ #
    selected_indices, performance_list = progressive_feature_selection(
        X_train_norm,
        y_train,
        ranked_indices
    )

    best_fitness = max(f for _, f in performance_list)
    num_features_to_select = max(
        n for n, f in performance_list if f == best_fitness
    )

    selected_indices_final = selected_indices[:num_features_to_select]
    selected_feature_names = [
        str(feature_names[i]) for i in selected_indices_final
    ]

    print("\nSelected feature names after progressive selection:")
    for f in selected_feature_names:
        print(" -", f)

    print("\nPerformance at each step:")
    for n_features, fitness in performance_list:
        print(f"{n_features:2d} features -> fitness: {fitness:.3f}")

    # ------------------------ Plot Performance ------------------------ #
    num_features_list = [x[0] for x in performance_list]
    fitness_list = [x[1] for x in performance_list]

    plt.figure(figsize=(8, 4))
    plt.plot(num_features_list, fitness_list, marker="o")
    plt.xlabel("Number of Features Selected")
    plt.ylabel("Fitness")
    plt.title("Progressive Feature Selection Performance")
    plt.grid(True)
    plt.savefig("progressive_selection.png", dpi=300)
    plt.show()

    # ------------------------ Final Datasets ------------------------ #
    X_train_sel = X_train_norm[:, selected_indices_final]
    X_val_sel = X_val_norm[:, selected_indices_final]

# ------------------------ Final Dataset Info ------------------------ #
print("\nFinal X_train shape:", X_train_sel.shape)
print("Final X_val shape:", X_val_sel.shape)

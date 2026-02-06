import numpy as np

from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from data_utils import select_features, track_indices


#----------------------- Accuracy -----------------------#

def compute_accuracy(y_true, y_pred):
    """
    Compute classification accuracy.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
    Returns:
        accurancy (float)
    """
    return accuracy_score(y_true, y_pred)

# ----------------------- Penalization -----------------------#

def compute_penalty(binary_vector, alpha):
    """
    Compute penalization term based on number of selected features.

    Args:
        binary_vector (array-like): Binary vector indicating selected features.
        alpha (float): Penalization coefficient.
    Returns:
        penalty (float)
    """
    num_selected_features = np.sum(binary_vector)
    total_features = len(binary_vector)

    return alpha * (num_selected_features / total_features)

#----------------------- Single fold evaluation -----------------------#

def evaluate_fold(X_train, X_val, y_train, y_vald):
    """
    Train and evaluate a logistic regression model on a single fold.
    Normalization is performed ONLY using training data to avaid Data Leakage.)

    Returns: 
        accuracy (float)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    model = LogisticRegression(
        max_iter=1000,
        solver='liblinear'    
    )
    
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_val_scaled)

    return compute_accuracy(y_vald, y_pred)

#----------------------- Fitness computation -----------------------#

def compute_fitness(
    individual,
    X,
    y,
    k_folds=5,
    alpha=0.001,
    random_state=None
):
    """
    Compute fitness of an individual using k-fold cross-validation.

    Args:
        individual (array-like): Binary vector representing selected features.
        X (DataFrame or ndarray): Feature dataset.
        y (array-like): Target labels.
        k_folds (int): Number of folds for cross-validation.
        alpha (float): Penalization coefficient.
        random_state (int or None): Random seed for reproducibility.
    Returns:
        fitness (float)
    """
    # If no features selected
    if np.sum(individual) == 0:
        return 0.0      
    
    # Select features once (indices reused across folds)
    X_reduced, _ = select_features(X, individual) 
    # Stores only the reduced features subset

    kf = KFold(
        n_splits=k_folds, 
        shuffle=True, 
        random_state=random_state 
        # random seed for shuffling
    )

    fold_fitness = []

    for train_idx, val_idx in kf.split(X_reduced):
        X_train, X_val = X_reduced[train_idx], X_reduced[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        accurancy = evaluate_fold(X_train, X_val, y_train, y_val)
        penalty = compute_penalty(individual, alpha)

        fold_fitness.append(accurancy - penalty)
    
    return np.mean(fold_fitness)

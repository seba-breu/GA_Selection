from sklearn.datasets import load_breast_cancer 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

#----------------------- Load data and preprocessing -----------------------#

def load_breast_cancer_data(as_dataframe=False):
    """
    Load the breast cancer dataset from sklearn and return features and target.
    
    Parameters:
    as_dataframe (bool): If True, returns features as a pandas DataFrame. 
                         If False, returns features as a numpy array.
    
    Returns:
    X: Features (DataFrame or ndarray)
    y: Target (ndarray)
    """
    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = data.feature_names

    if as_dataframe:
        X = pd.DataFrame(X, columns=data.feature_names)
        y = pd.Series(y)
    
    return X, y, feature_names

def train_val_split(X, y, test_size=0.2, random_state=42, stratify=True):
    """
    Split the breast cancer dataset into training and validation sets.
    
    Args:
        X: Features
        y: Labels
        test_size (float): Proportion of the dataset to include in the validation split.
        random_state (int or None): Random seed for reproducibility.
        stratify (bool): If True, perform stratified splitting based on labels.
        
    Returns:
        X_train, X_val, y_train, y_val"""
    
    if stratify:
        stratify_labels = y
    else:
        stratify_labels = None
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_labels
    )
    return X_train, X_val, y_train, y_val

def normalize_features(X_train, X_val=None):
    """
    Normalize features using StandardScaler (mean=0, std=1).
    
    Args:
        X_train: Training features (DataFrame or ndarray)
        X_val: Validation features (DataFrame or ndarray, optional)
    
    Returns:
        X_train_scaled, X_val_scaled (if X_val is provided) or X_train_scaled only"""
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    if X_val is not None:
        X_val_scaled = scaler.transform(X_val)
        return X_train_scaled, X_val_scaled
    return X_train_scaled   

# Ref.: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html

#----------------------- Generate GA population -----------------------#

def initialize_binary_vector(pop_size, num_features, random_state=None):
    """
    Initialize a population of binary vectors.
    
    Parameters:
    pop_size (int): Number of binary vectors to generate (size of the GENETIC POPULATION).
    num_features (int): Length of each binary vector.
    random_state (int or None): Random seed for reproducibility.
    
    Returns:
    population (ndarray): Array of shape (pop_size, num_features) with binary values.
    """
    rng = np.random.default_rng(random_state) # Random number generator
    population = rng.integers(0, 2, size=(pop_size, num_features)) 
    # 0 included to 2 excluded which means -> Generate binary values in {0, 1}

    return population

#----------------------- Feature selection based on binary vector -----------------------#

def select_features(X, binary_vector):
    """
    Select features from the dataset based on the binary vector.
    
    Parameters:
    X: Features (DataFrame or ndarray)
    binary_vector (ndarray): Binary vector indicating which features to select.
    
    Returns:
    Reduced subset of features selected based on the binary vector and their indices.
    """
    selected_indices = np.where(binary_vector == 1)[0]
    return X[:, selected_indices], selected_indices

def track_indices(binary_vector):
    """
    Track the original indices of selected features based on the binary vector.
    
    Args:
        binary_vector (array-like)
    
    Returns:
        indices (np.ndarray)
        
    """
    return np.where(binary_vector == 1)[0]
    
    # unlike select_features, this function only returns the indices of selected features

#----------------------- Track & map selected feature indices and names -----------------------#

def get_selected_feature_names(feature_names, binary_vector):
    """
    Map the selected indices from the binary vector to the original feature names.
    
    Args: 
        feature_names (array-like)
        binary_vector (array-like)

    Returns:
        selected_feature_names (list)

    """
    return [feature_names[i] for i in track_indices(binary_vector)]






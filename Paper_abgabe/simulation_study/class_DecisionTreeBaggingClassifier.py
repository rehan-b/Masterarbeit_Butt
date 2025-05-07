import numpy as np
from sklearn.tree import DecisionTreeClassifier


class DecisionTreeBaggingClassifier:
    """
    A bagging classifier that uses multiple DecisionTreeClassifiers with optional
    weighted bootstrapping. The classifier supports binary classification, where
    the class labels are expected to be 0 and 1. The predicted probabilities
    returned by the `predict_proba` method correspond to the probability of class 1.
    
    Parameters:
    -----------
    params : dict
        A dictionary containing the following parameters for the decision tree
        and bagging process:
        - 'n_estimators' : int
            Number of decision trees to use in the bagging ensemble.
        - 'max_depth' : int
            The maximum depth of each decision tree.
        - 'min_samples_split' : int
            The minimum number of samples required to split an internal node.
        - 'max_features' : str or int, optional
            The number of features to consider when looking for the best split.
        - 'random_state' : int
            Random seed used for bootstrapping and decision tree construction.
        - 'weighted_bootstrapping' : bool
            If True, the classifier uses weighted bootstrapping. Sample weights can
            be provided during fitting.

    Methods:
    --------
    create_bootstrap_indices_and_Nbi(n, B, weights=None)
        Generate bootstrap sample indices and count occurrences (Nbi) for each sample.
        
    fit(X, y, sample_weights=None)
        Train the ensemble of decision trees using bootstrapped datasets from X and y.
        Optionally uses weighted bootstrapping if sample_weights are provided.
        
    predict_proba(X)
        Predict the probability of class 1 for each sample in X. Returns both the
        per-tree probabilities and the averaged probabilities across all trees.
        
    set_random_state(random_state)
        Update the random state used for bootstrapping and decision tree initialization.

    Notes:
    ------
    - This classifier is designed for **binary classification**, where class labels
      are restricted to `0` and `1`.
    - The `predict_proba` method returns the probability for class `1` only.
    - The class uses bagging (bootstrap aggregating) to train multiple decision trees
      on different bootstrapped samples, which may or may not include weighted
      sampling if specified.
    """

    def __init__(self, params: dict):
        """
        Initialize the DecisionTreeBaggingClassifier with the provided parameters.

        Parameters:
        -----------
        params : dict
            Dictionary containing parameters for the decision trees and bagging process.
            Must include 'n_estimators', and can optionally include 'max_depth', 
            'min_samples_split', 'max_features', 'random_state', and 'weighted_bootstrapping'.
        """
        self.n_trees = params.get('n_estimators')
        self.weighted_bootstrapping = params.get('weighted_bootstrapping')
        self.tree_params = params
        self.trees_ = []

    def create_bootstrap_indices_and_Nbi(
        self, n: int, B: int, weights: np.ndarray = None
    ):
        """
        Generate bootstrap sample indices and count occurrences (Nbi) for each sample.

        Parameters:
        -----------
        n : int
            The number of samples in the dataset.
        B : int
            The number of bootstrap samples (equal to the number of trees).
        weights : np.ndarray, optional
            Weights for weighted bootstrapping. If None, standard bootstrapping is used.

        Returns:
        --------
        boot_indices : np.ndarray
            Indices for the bootstrapped samples.
        nbi : np.ndarray
            The number of times each sample appears in each bootstrap sample.
        """
        if weights is None:
            # Standard bootstrapping
            rng = np.random.default_rng(self.tree_params.get('random_state'))
            boot_indices = rng.choice(np.arange(n), size=(B, n), replace=True)
        else:
            # Weighted bootstrapping
            rng = np.random.default_rng(self.tree_params.get('random_state'))
            boot_indices = rng.choice(np.arange(n), size=(B, n), p=weights, replace=True)
        
        # Count occurrences of each sample in each bootstrap sample
        nbi = np.apply_along_axis(lambda x: np.bincount(x, minlength=n), axis=1, arr=boot_indices)
        return boot_indices, nbi

    def fit(self, X, y, sample_weights=None):
        """
        Train the ensemble of decision trees using bootstrapped samples from X and y.

        Parameters:
        -----------
        X : np.ndarray
            Feature matrix for the training data.
        y : np.ndarray
            Labels (class 0 or class 1) for the training data.
        sample_weights : np.ndarray, optional
            Weights for each sample, used for weighted bootstrapping if specified.

        Returns:
        --------
        self : DecisionTreeBaggingClassifier
            The trained model.
        """
        n_samples = X.shape[0]

        # Generate bootstrap indices and counts
        if self.weighted_bootstrapping and sample_weights is not None:
            self.boot_indices, self.nbi = self.create_bootstrap_indices_and_Nbi(
                n=n_samples, B=self.n_trees, weights=sample_weights
            )
        else:
            self.boot_indices, self.nbi = self.create_bootstrap_indices_and_Nbi(
                n=n_samples, B=self.n_trees
            )
        
        # Train a decision tree for each bootstrap sample
        for i in range(self.n_trees):
            X_resampled, y_resampled = X[self.boot_indices[i]], y[self.boot_indices[i]]
            
            # Create and fit a decision tree with the specified parameters
            tree = DecisionTreeClassifier(
                max_depth=self.tree_params.get('max_depth'),
                min_samples_split=self.tree_params.get('min_samples_split'),
                max_features=self.tree_params.get('max_features'),
                random_state=self.tree_params.get('random_state')
            )
            
            tree.fit(X_resampled, y_resampled)
            self.trees_.append(tree)
        
        return self

    def predict_proba(self, X):
        """
        Predict class probabilities for class 1 using the ensemble of decision trees.

        Parameters:
        -----------
        X : np.ndarray
            Feature matrix for the test data.

        Returns:
        --------
        preds_trees : np.ndarray
            The predicted probabilities for class 1 from each individual decision tree.
        pred_ : np.ndarray
            The averaged predicted probabilities for class 1 across all trees.
        """
        n_samples = X.shape[0]
        preds_trees = np.zeros((self.n_trees, n_samples))
        
        for b, tree in enumerate(self.trees_):
            proba = tree.predict_proba(X)
            
            # Get the probability for class 1 (assuming binary classification with labels 0 and 1)
            if len(tree.classes_) == 2:
                preds_trees[b] = proba[:, 1]
            else:
                # If only one class is present in the bootstrap sample
                preds_trees[b] = np.ones(n_samples) if tree.classes_[0] == 1 else np.zeros(n_samples)
        
        # Average the probabilities across all trees
        pred_ = np.mean(preds_trees, axis=0)

        return preds_trees, pred_

    def set_random_state(self, random_state):
        """
        Update the random_state in tree_params.

        Parameters:
        -----------
        random_state : int
            The new random state to be used for bootstrapping and decision trees.
        """
        self.tree_params['random_state'] = random_state
        #print(f"Updated random_state to: {self.tree_params['random_state']}")

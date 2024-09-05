from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state
from sklearn.base import clone
import numpy as np
from sklearn.utils.multiclass import unique_labels

'''Class Documentation: CustomRandomForestClassifier
This class is an extension of the RandomForestClassifier from scikit-learn. 
The primary difference is that it allows users to pass custom samples for training each decision tree in the forest. 
Normally, a RandomForestClassifier uses bootstrapping (random sampling with replacement) for each tree,
but this version allows explicit control over which samples are used for each tree.'''

class CustomRandomForestClassifier(RandomForestClassifier):
    def __init__(self, custom_samples=None, **kwargs):
        super().__init__(**kwargs)
        self.custom_samples = custom_samples  # User-defined samples for each tree
        self.custom_estimators_samples_ = []  # Store the actual samples used in each tree
        self.estimator_ = DecisionTreeClassifier()  # Define the base estimator (DecisionTreeClassifier)

    def fit(self, X, y, sample_weight=None):
        """
        Overwrites the fit method of RandomForestClassifier to use user-defined samples for training each tree.
        If custom samples are provided, each tree will be trained on the respective subset of data.
        Otherwise, the standard bootstrapping method of RandomForestClassifier will be used.
        """
        # Set attributes related to classes, as done in the base RandomForestClassifier
        self.n_classes_ = np.unique(y).shape[0]  # Number of classes
        self.classes_ = unique_labels(y)  # Unique classes
        self.n_outputs_ = 1  # For single-output classification

        # Check if custom samples have been provided
        if self.custom_samples is not None:
            random_state = check_random_state(self.random_state)
            
            # Initialize the list to store trained trees and used samples
            self.estimators_ = []
            self.custom_estimators_samples_ = []  # Initialize list for custom samples used in each tree
            
            for i in range(self.n_estimators):
                # Use the user-defined samples for the current tree
                sample_indices = self.custom_samples[i]
                
                # Clone the base estimator to ensure each tree is independent
                tree = clone(self.estimator_)
                tree.set_params(random_state=random_state.randint(np.iinfo(np.int32).max))  # Set random state for reproducibility
                tree.fit(X[sample_indices], y[sample_indices])  # Train the tree on the custom samples
                
                # Save the trained tree and the sample indices used
                self.estimators_.append(tree)
                self.custom_estimators_samples_.append(sample_indices)

        else:
            # If no custom samples are provided, use the default RandomForestClassifier behavior
            super().fit(X, y, sample_weight=sample_weight)
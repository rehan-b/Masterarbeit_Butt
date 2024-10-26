import numpy as np
from sklearn.tree import DecisionTreeClassifier
from joblib import Parallel, delayed


class DecisionTreeBaggingClassifier:
    """
    [Die ursprüngliche Klassendokumentation bleibt unverändert]
    """

    def __init__(self, params: dict = None):
        if params is None:
            params = {}
        self.n_estimators = params.get('n_estimators', 100)  # Default-Wert
        self.weighted_bootstrapping = params.get('weighted_bootstrapping', False)  # Default-Wert
        self.tree_params = params
        self.trees_ = []
        self.n_jobs = params.get('n_jobs', -1)  # Anzahl der parallelen Jobs (Standard: alle verfügbaren Kerne)

    def create_bootstrap_indices_and_Nbi(
        self, n: int, B: int, weights: np.ndarray = None
    ):
        """
        [Die Methode bleibt unverändert]
        """
        if weights is None:
            # Standard-Bootstrapping
            rng = np.random.default_rng(self.tree_params.get('random_state'))
            boot_indices = rng.choice(np.arange(n), size=(B, n), replace=True)
        else:
            # Gewichtetes Bootstrapping
            rng = np.random.default_rng(self.tree_params.get('random_state'))
            boot_indices = rng.choice(np.arange(n), size=(B, n), p=weights, replace=True)
        
        # Zähle die Häufigkeit jedes Samples in jedem Bootstrap-Sample
        nbi = np.apply_along_axis(lambda x: np.bincount(x, minlength=n), axis=1, arr=boot_indices)
        return boot_indices, nbi

    def fit(self, X, y, sample_weights=None):
        """
        Trainiert das Ensemble von Entscheidungsbäumen mit paralleler Verarbeitung.
        """
        n_samples = X.shape[0]

        # Generiere Bootstrap-Indizes und Zählungen
        if self.weighted_bootstrapping and sample_weights is not None:
            self.boot_indices, self.nbi = self.create_bootstrap_indices_and_Nbi(
                n=n_samples, B=self.n_estimators, weights=sample_weights
            )
        else:
            self.boot_indices, self.nbi = self.create_bootstrap_indices_and_Nbi(
                n=n_samples, B=self.n_estimators
            )
        
        # Definiere eine Funktion zum Trainieren eines einzelnen Baums
        def train_single_tree(i):
            X_resampled, y_resampled = X[self.boot_indices[i]], y[self.boot_indices[i]]
            
            # Erstelle und trainiere einen Entscheidungsbaum mit den angegebenen Parametern
            tree = DecisionTreeClassifier(
                max_depth=self.tree_params.get('max_depth'),
                min_samples_split=self.tree_params.get('min_samples_split'),
                max_features=self.tree_params.get('max_features'),
                random_state=self.tree_params.get('random_state') + i if self.tree_params.get('random_state') is not None else None
            )
            
            tree.fit(X_resampled, y_resampled)
            return tree

        # Trainiere die Bäume parallel
        self.trees_ = Parallel(n_jobs=self.n_jobs)(
            delayed(train_single_tree)(i) for i in range(self.n_estimators)
        )
        
        return self

    def predict_proba(self, X):
        """
        [Die Methode bleibt unverändert]
        """
        n_samples = X.shape[0]
        preds_trees = np.zeros((self.n_estimators, n_samples))
        
        for b, tree in enumerate(self.trees_):
            proba = tree.predict_proba(X)
            
            # Hole die Wahrscheinlichkeit für Klasse 1 (bei binärer Klassifikation mit Labels 0 und 1)
            if len(tree.classes_) == 2:
                preds_trees[b] = proba[:, 1]
            else:
                # Falls nur eine Klasse im Bootstrap-Sample vorhanden ist
                preds_trees[b] = np.ones(n_samples) if tree.classes_[0] == 1 else np.zeros(n_samples)
        
        # Mittlere Wahrscheinlichkeit über alle Bäume
        pred_ = np.mean(preds_trees, axis=0)

        return preds_trees, pred_

    def set_random_state(self, random_state):
        """
        [Die Methode bleibt unverändert]
        """
        self.tree_params['random_state'] = random_state
        #print(f"Updated random_state to: {self.tree_params['random_state']}")
        
    def get_params(self, deep=True):
        """
        [Die Methode bleibt unverändert]
        """
        return self.tree_params
    
    def set_params(self, **params):
        """
        Setzt die Parameter für den Classifier.
        """
        for key, value in params.items():
            if key in self.tree_params:
                self.tree_params[key] = value
            elif key == "n_estimators":
                self.n_estimators = value
            elif key == "weighted_bootstrapping":
                self.weighted_bootstrapping = value
        return self

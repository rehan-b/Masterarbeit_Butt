import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines import KaplanMeierFitter, WeibullAFTFitter
from sksurv.metrics import concordance_index_ipcw
from sksurv.util import Surv
from sklearn.model_selection import train_test_split
from class_DecisionTreeBaggingClassifier import DecisionTreeBaggingClassifier


def create_new_dataset_with_ipcw_weights(
    data: pd.DataFrame, t: np.float64, kmf=None
) -> pd.DataFrame:
    """
    Create a new dataset with inverse probability of censoring weighting (IPCW) weights.
    """
    if kmf is None:
        # Fit the Kaplan-Meier estimator
        kmf = KaplanMeierFitter()
        kmf.fit(
            durations=data["time"].astype(float),
            event_observed=1 - data["event"].astype(bool),
        )

    # Copy the data to avoid modifying the original dataframe
    new_data = data.copy()

    # Determine survival status at time t
    conditions = [
        (new_data["time"] <= t) & (new_data["event"] == 1),  # Died before t
        (new_data["time"] >= t),                             # Survived past t
        (new_data["time"] <= t) & (new_data["event"] == 0),  # Censored before t
    ]
    choices = [0, 1, 999]
    new_data["survived"] = np.select(conditions, choices, default=999)

    # Calculate IPCW weights
    survival_times = new_data["time"]
    survival_probabilities = kmf.survival_function_at_times(survival_times).values.flatten()
    ipcw_weights = 1 / survival_probabilities
    ipcw_weight_tau = 1 / kmf.survival_function_at_times(t).values.flatten()[0]

    # Assign weights
    new_data["weights_ipcw"] = np.where(
        new_data["survived"] == 1,
        ipcw_weight_tau,
        np.where(new_data["survived"] == 0, ipcw_weights, 0),
    )

    # Normalize weights
    new_data["weights_ipcw"] /= new_data["weights_ipcw"].sum()

   
    portions_at_cutpoint = new_data["survived"].value_counts(normalize=True)
    if 999 in portions_at_cutpoint.keys():
        portion_censored_after_cut = portions_at_cutpoint[999]
    else:
        portion_censored_after_cut = 0

    if 0 in portions_at_cutpoint.keys():
        n_events_after_cut = portions_at_cutpoint[0] * new_data.shape[0]
    else:
        n_events_after_cut = 0

    return (
        new_data,
        n_events_after_cut,
        portion_censored_after_cut,

    )


def ipc_weighted_mse(y_true, y_pred, sample_weight):
    """
    Calculates the weighted mean squared error (MSE) between the true values and the predicted values.

    Parameters:
    - y_true (array-like): The true values.
    - y_pred (array-like): The predicted values.
    - sample_weight (array-like): The weights assigned to each sample.

    Returns:
    - weighted_mse (float): The weighted mean squared error.

    """
    return np.average((y_true - y_pred) ** 2, weights=sample_weight)

def calculate_ijk_variance(
    clf: DecisionTreeBaggingClassifier, X_pred_point: pd.DataFrame, df_train: pd.DataFrame
) -> float:
    """
    Calculates the biased variance estimate and bias correction for a given random forest classifier,
    prediction point, and training data.
    Parameters:
    - clf: The classifier object used for prediction.
    - X_pred_point: The prediction point as a pandas DataFrame.
    - df_train: The training data as a pandas DataFrame.
    Returns:
    - biased_var_estimate: The biased variance estimate.
    - bias_correction: The bias correction.
    """

    T_N_b, pred = clf.predict_proba(X_pred_point.values)
    N_bi = clf.nbi
    weights = df_train["weights_ipcw"]
    B, n = N_bi.shape
    T_N_b_mean = np.mean(T_N_b, axis=0)

    cov_i = ((N_bi - n * weights.values.reshape(1,-1)).T @ (T_N_b - T_N_b_mean)) / B
    cov_i_hoch2 = cov_i**2
    array = cov_i_hoch2/weights.values.reshape(-1,1)

    biased_var_estimate = np.sum(array[~np.isnan(array) & ~np.isinf(array)], axis=0) * np.sum(weights**2)

    bias_correction = n / B * np.sum(1-weights[weights > 0]) * np.var(T_N_b, axis=0, ddof=1)* np.sum(weights**2)

    return biased_var_estimate, bias_correction

def calculate_bootstrap_variance(
    X_pred_point: pd.DataFrame,
    params_rf: dict,
    df_train: pd.DataFrame,
    seed: int,
    B_first_level: int,
    tau: np.float64,
) -> float:
    """
    Calculate the bootstrap variance of predictions using random forest classifier.
    Parameters:
        X_pred_point (pd.DataFrame): The input data for prediction.
        params_rf (dict): The parameters for the random forest classifier.
        df_train (pd.DataFrame): The training dataset.
        seed (int): The seed for random number generation.
        B_first_level (int): The number of bootstrap samples.
        tau (np.float64): The time point for IPCW weights.
    Returns:
        float: The bootstrap variance of predictions.
    """

    np_train = df_train.values
    df_train_columns_name = df_train.columns
    preds = np.empty(B_first_level)

    # Generate firstlevel bootstrapped indices
    df = create_new_dataset_with_ipcw_weights(data=df_train, t=tau)
    
    
    rng = np.random.default_rng(seed)
    first_level_boot_indices = rng.choice(
        a=np.arange(df_train.shape[0]), size=(B_first_level, df_train.shape[0]), replace=True
    )
    
    for b in range(B_first_level):

        np_train_boot = np_train[first_level_boot_indices[b], :]

        # Create the new dataset with IPCW weights
        df_train_boot = create_new_dataset_with_ipcw_weights(
            data=pd.DataFrame(np_train_boot, columns=df_train_columns_name), t=tau
        )

        # Set the random state and fit the classifier
        clf = DecisionTreeBaggingClassifier(params_rf)
        clf.set_random_state(random_state=seed + 1000+ b )
        
        clf.fit(
            X=df_train_boot.drop(
                ["time", "event", "weights_ipcw", "survived"], axis=1
            ).values,
            y=df_train_boot["survived"].values,
            sample_weights=df_train_boot["weights_ipcw"].values,
        )
        
        # Predict and store result
        _ ,pred = clf.predict_proba(X_pred_point.values)
        preds[b] = pred[0]

    # Calculate variance using numpy
    return np.var(preds, ddof=1)

def calculate_jk_after_bootstrap_variance(
    clf: DecisionTreeBaggingClassifier,
    X_pred_point: pd.DataFrame,
    params_rf: dict,
    df_train: pd.DataFrame,
) -> float:
    """
    Calculates the Jackknife-after-Bootstrap variance (unbiased, if equal weights are used during bootstrapsampling)
    for a given random forest classifier.
    Parameters:
        clf (DecisionTreeBaggingClassifier): The random forest classifier.
        X_pred_point (pd.DataFrame): The input data point for prediction.
        params_rf (dict): The parameters of the random forest.
        df_train (pd.DataFrame): The training dataset.
    Returns:
        float: The Jackknife-after-Bootstrap variance.
    """
    n_samples = df_train.shape[0]

    # Precompute predictions for all trees
    tree_preds, theta = clf.predict_proba(X_pred_point.values)

    # Cache the estimators' samples array for efficient reuse
    estimators_samples = clf.boot_indices

    # Prepare a boolean mask for each sample's presence in each estimator's bootstrap
    presence_mask = np.zeros((n_samples, params_rf["n_estimators"]), dtype=bool)
    for i, samples in enumerate(estimators_samples):
        samples = np.array(samples, dtype=int)
        presence_mask[samples, i] = True

    theta_is = []
    for ii in range(n_samples):
        indices_without_ii = np.where(~presence_mask[ii])[0]
        if 0 < len(indices_without_ii) < params_rf["n_estimators"]:
            theta_is.append(tree_preds[indices_without_ii].mean())

    theta_is = np.array(theta_is)
    var_jka_biased = np.sum((theta_is - theta) ** 2) * (n_samples - 1) / n_samples

    var_jka_correction = (
        (np.exp(1) - 1)
        * (n_samples / params_rf["n_estimators"])
        * np.var(tree_preds, ddof=1)
    )
    return var_jka_biased - var_jka_correction


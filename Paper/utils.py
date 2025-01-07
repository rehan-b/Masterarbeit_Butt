import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines import KaplanMeierFitter, WeibullAFTFitter
from sksurv.util import Surv
from sklearn.model_selection import train_test_split
from class_DecisionTreeBaggingClassifier import DecisionTreeBaggingClassifier


def calculate_ijk_butt_variance(
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

    return biased_var_estimate-bias_correction

def calculate_ijk2_butt_variance(
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
    array = cov_i_hoch2/((weights.values.reshape(-1,1))**2)

    biased_var_estimate = np.sum(array[~np.isnan(array) & ~np.isinf(array)], axis=0) * (1/(np.sum(weights > 0))**2)

    bias_correction = n / B * (1/np.sum(weights > 0)**2) * np.var(T_N_b, axis=0, ddof=1)* np.sum(1/(weights[weights > 0]) -1)

    return biased_var_estimate-bias_correction

def calculate_ijk_wager_variance(
    clf: DecisionTreeBaggingClassifier, X_pred_point: pd.DataFrame, df_train: pd.DataFrame
) -> float:

    T_N_b, pred = clf.predict_proba(X_pred_point.values)
    N_bi = clf.nbi
    B, n = N_bi.shape

    cov_i = ((N_bi - 1).T @ (T_N_b - pred)) / B
    cov_i_hoch2 = cov_i**2

    biased_var_estimate = np.sum(cov_i_hoch2) 

    bias_correction = n / B * np.var(T_N_b, axis=0, ddof=1)

    return biased_var_estimate-bias_correction


def calculate_bootstrap_variance(
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



#########################################################################################################

def create_weibull_data(params: dict, random_state: int) -> pd.DataFrame:

    (shape_weibull, scale_weibull_base, rate_censoring, n, p_1, p_2, p_3, p_4) = (
        params["shape_weibull"],
        params["scale_weibull_base"],
        params["rate_censoring"],
        params["n"],
        params["p_1"], 
        params["p_2"],
        params["p_3"],
        params["p_4"], )

    # Kovariaten
    rng = np.random.default_rng(random_state)
    X_1 = rng.binomial(1, 0.3, n)
    X_2 = rng.binomial(1, 0.8, n)
    X_3 = rng.normal(80, 10, n)  
    X_4 = rng.normal(40, 5, n)

    # Lambda Weibull
    lambda_weibull = scale_weibull_base * np.exp(
          p_1 * X_1
        + p_2 * X_2  
        + p_3 * X_3 
        + p_4 * X_4 
    )

    # Ereigniszeiten und Zensierzeiten
    event_times = rng.weibull(shape_weibull, n) * lambda_weibull
    censoring_times = rng.exponential(1 / rate_censoring, n)
    observed_times = np.minimum(event_times, censoring_times)
    event_occurred = event_times <= censoring_times

    # Erstellung des Datensatzes
    data = pd.DataFrame(
        {
            "X_1": X_1.astype(int),
            "X_2": X_2.astype(int),
            "X_3": X_3,
            "X_4": X_4,
            "time": observed_times,
            "event": event_occurred.astype(int),
        }
    )

    return data 

def stratified_split(data: pd.DataFrame, random_state: int, test_size = 0.3 ) -> pd.DataFrame:

    ## Startified Split in Train und Testdaten
    X = data[["X_1", "X_2", "X_3","X_4"]]
    y = Surv.from_arrays(event=data["event"], time=data["time"])
    df_train, df_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=data["event"]
    )

    # Transform to DataFrame
    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)

    y_train_df = pd.DataFrame(y_train, columns=["event", "time"])
    y_test_df  = pd.DataFrame(y_test, columns=["event", "time"])

    df_train[["event", "time"]] = y_train_df[["event", "time"]]
    df_test[["event", "time"]] = y_test_df[["event", "time"]]

    return df_train, df_test

def create_data_with_ipc_weights(params: dict, data: pd.DataFrame) -> pd.DataFrame:

    tau = params["tau"]
    # Fit the Kaplan-Meier estimator
    kmf = KaplanMeierFitter()
    kmf.fit(np.array(data["time"],dtype=float), event_observed=  np.array(1 - data["event"],dtype=bool))

    # Efficiently calculate the 'survived' column using np.select for vectorized operations
    conditions = [
        (data["time"] <= tau) & (data["event"] == 1),
        (data["time"] >= tau),
        (data["time"] <= tau) & (data["event"] == 0),
    ]
    choices = [0, 1, 999]
    data["survived"] = np.select(conditions, choices, default=999)

    # Calculate the IPCW weights
    survival_times = data["time"]
    survival_probabilities = kmf.survival_function_at_times(
        survival_times
    ).values.flatten()
    ipcw_weights = 1 / survival_probabilities
    ipcw_weight_tau = 1 / kmf.survival_function_at_times(tau).values.flatten()[0]

    # Apply weights based on the 'survived' column
    data["weights_ipcw"] = np.where(
        data["survived"] == 1,
        ipcw_weight_tau,
        np.where(data["survived"] == 0, ipcw_weights, 0),
    )

    # Normalize the weights
    data["weights_ipcw"] /= data["weights_ipcw"].sum()

    return data

def ipc_weighted_mse(y_true, y_pred, sample_weight):

    return np.average((y_true - y_pred) ** 2, weights=sample_weight)

def calculate_true_survival_probability(individual, params):
    (
        shape_weibull,
        scale_weibull_base,
        p_1,
        p_2,
        p_3,
        p_4,
    ) = (
        params["shape_weibull"],
        params["scale_weibull_base"],
        params["p_1"],
        params["p_2"],
        params["p_3"],
        params["p_4"],
    )

    # Extrahieren der Kovariaten
    X_1 = individual['X_1'].values[0]
    X_2 = individual['X_2'].values[0]
    X_3 = individual['X_3'].values[0]
    X_4 = individual['X_4'].values[0]

    # Berechnung des linearen Prädiktors (LP)
    LP =  X_1 * p_1 +X_2 * p_2 + X_3 * p_3 +X_4 * p_4 

    # Individueller Skalenparameter lambda
    lambda_weibull = scale_weibull_base * np.exp(LP)

    # Überlebensfunktion
    S_t = np.exp(- (params["tau"] / lambda_weibull) ** shape_weibull)

    return S_t

#########################################################################################################

def simulation(
    seed: int,
    data_generation_parameter: dict,
    params_rf: dict,
    train_models: bool ,
    ijk_wager_calc: bool,
    ijk_butt_calc: bool,
    ijk2_butt_calc: bool, 
    boot_calc: bool,
    B_first_level: int ,
    X_pred_point : pd.DataFrame,):

    if data_generation_parameter['data_type'] == 'weibull':
        data = create_weibull_data(params=data_generation_parameter, random_state=seed)
        df_train, df_test = stratified_split(data=data, random_state=seed, test_size=0.3)
        df_train = create_data_with_ipc_weights(data=df_train, params=data_generation_parameter)
        df_test = create_data_with_ipc_weights(data=df_test, params=data_generation_parameter)
        portion_zero_weights_train = df_train['weights_ipcw'].value_counts(normalize=True)[0]


    ### Weibull Modell ####
    if train_models:

        # Fit
        aft = WeibullAFTFitter()
        aft.fit(
            df=df_train.drop(["weights_ipcw", "survived"], axis=1),
            duration_col="time",
            event_col="event",
        )

        # Evaluation auf Testdaten
        y_pred = (
            aft.predict_survival_function(
                df=df_test.drop(["weights_ipcw", "survived", "time", "event"], axis=1),
                times=data_generation_parameter["tau"],
            )
            .iloc[0]
            .tolist()
        )
        wb_test_mse = ipc_weighted_mse(
            y_true=df_test["survived"].values,
            y_pred=y_pred,
            sample_weight=df_test["weights_ipcw"],
        )

        # Prediction für X_erwartung
        wb_pred = (
            aft.predict_survival_function(df=X_pred_point, times=data_generation_parameter["tau"]).iloc[0].tolist()
        )
    

        ### Random Forest Modell ###
        # Fit
        params_rf["random_state"] = seed
        clf = DecisionTreeBaggingClassifier(params_rf)
        clf.fit(
            X=df_train.drop(
                ["time", "event", "weights_ipcw", "survived"], axis=1
            ).values,
            y=df_train["survived"].values,
            sample_weights=df_train["weights_ipcw"].values,
        )

        # Evaluation auf Testdaten
        _ , pred  =clf.predict_proba(df_test.drop(
            ["weights_ipcw", "survived", "time", "event"], axis=1
        ).values)
        rf_test_mse = ipc_weighted_mse(
            y_true=df_test["survived"].values,
            y_pred=pred,
            sample_weight=df_test["weights_ipcw"].values,
        )

        # Prediction für X_erwartung
        _ ,rf_pred = clf.predict_proba(X_pred_point.values)

    else:
        wb_test_mse = 0.
        rf_test_mse = 0.0
        wb_pred = [0.0]
        rf_pred = [0.0]

    #### Variance Estimation ####
    ### Wager

    ### Butt
    if ijk_butt_calc:
        ijk_butt_var = calculate_ijk_butt_variance(
            clf=clf, X_pred_point=X_pred_point, df_train=df_train
        )
    else:
        ijk_butt_var = 0.0

    ### Butt2
    if ijk2_butt_calc:
        ijk2_butt_var =   calculate_ijk2_butt_variance(
            clf=clf, X_pred_point=X_pred_point, df_train=df_train
        )
    else:
        ijk2_butt_var = 0.0

    if ijk_wager_calc:
        ijk_wager_var = calculate_ijk_wager_variance(
            clf=clf, X_pred_point=X_pred_point, df_train=df_train
        )
    else:
        ijk_wager_var = 0.0

    ### boot
    if boot_calc:
        boot_var = calculate_bootstrap_variance(
        clf=clf,
        X_pred_point = X_pred_point,
        params_rf = params_rf,
        df_train= df_train,
        )
    else:
        boot_var = 0.0

    return (
        wb_test_mse,
        rf_test_mse,
        wb_pred,
        rf_pred,
        ijk_wager_var,
        ijk_butt_var,
        ijk2_butt_var,
        boot_var,
        df_train['survived'].value_counts(normalize=True),
        df_test['survived'].value_counts(normalize=True),
        calculate_true_survival_probability(X_pred_point, data_generation_parameter),
    )

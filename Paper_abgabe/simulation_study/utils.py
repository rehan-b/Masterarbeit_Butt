import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter, WeibullAFTFitter
from sksurv.util import Surv
from sklearn.model_selection import train_test_split
from class_DecisionTreeBaggingClassifier import DecisionTreeBaggingClassifier
import os, json
import matplotlib.pyplot as plt
import warnings

#########################################################################################################
' Variance Estmators '
def calculate_ijk_butt_variance(
    clf: DecisionTreeBaggingClassifier, X_pred_point: pd.DataFrame, df_train: pd.DataFrame
) -> float:

    T_N_b, pred = clf.predict_proba(X_pred_point.values)
    N_bi = clf.nbi
    weights = df_train["weights_ipcw"]
    B, n = N_bi.shape

    cov_i = ((N_bi - n * weights.values.reshape(1,-1)).T @ (T_N_b - pred)) / B
    cov_i_hoch2 = cov_i**2
    array = cov_i_hoch2/weights.values.reshape(-1,1)

    biased_var_estimate = np.sum(array[~np.isnan(array) & ~np.isinf(array)], axis=0) * np.sum(weights**2)

    # bias correction 1
    bias_correction = n / B * np.sum(1-weights[weights > 0]) * np.var(T_N_b, axis=0, ddof=1)* np.sum(weights**2)

    # bias correction 2
    bb = np.var((N_bi - n * weights.values.reshape(1,-1)) * (T_N_b - pred), axis=0, ddof=1)  / weights.values.reshape(1,-1)
    bias_correction2 = np.sum(weights**2) * 1/B  * np.sum(bb[~np.isnan(bb) & ~np.isinf(bb)], axis=0)

    return biased_var_estimate , bias_correction[0], bias_correction2

def calculate_ijk_jahn_variance(
    clf: DecisionTreeBaggingClassifier, X_pred_point: pd.DataFrame, df_train: pd.DataFrame
) -> float:

    T_N_b, pred = clf.predict_proba(X_pred_point.values)
    N_bi = clf.nbi
    weights = df_train["weights_ipcw"]
    B, n = N_bi.shape
    n_plus = np.sum(weights > 0)

    cov_i = ((N_bi - n * weights.values.reshape(1,-1)).T @ (T_N_b - pred)) / B
    cov_i_hoch2 = cov_i**2
    array = cov_i_hoch2/((weights.values.reshape(-1,1))**2)

    biased_var_estimate = np.sum(array[~np.isnan(array) & ~np.isinf(array)], axis=0) * (1/(np.sum(weights > 0))**2)

    #bias_correction1
    bias_correction =  (1/n_plus**2)  * np.var(T_N_b, axis=0, ddof=1)* n / B * np.sum( ( 1 / (weights[weights > 0] ) ) -1) 
    
    # bias correction 2
    bb = np.var((N_bi - n * weights.values.reshape(1,-1)) * (T_N_b - pred), axis=0, ddof=1)  / (weights**2).values.reshape(1,-1)
    bias_correction2 = (1/n_plus**2) * 1/B  * np.sum(bb[~np.isnan(bb) & ~np.isinf(bb)], axis=0)

    return biased_var_estimate , bias_correction[0], bias_correction2

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

    return biased_var_estimate , bias_correction[0]

def calculate_jk_variance(
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
    return var_jka_biased , var_jka_correction


    return 0.0 , 0.0

def calculate_bootstrap_variance(
    params_rf: dict,
    df_train: pd.DataFrame,
    seed: int,
    B_1: int,
    data_generation_parameter: dict,
) -> float:

    np_train = df_train.values
    df_train_columns_name = df_train.columns
    preds = np.empty(B_1)
    
    rng = np.random.default_rng(seed)
    first_level_boot_indices = rng.choice(
        a=np.arange(df_train.shape[0]), size=(B_1, df_train.shape[0]), replace=True
    )
    
    for b in range(B_1):

        np_train_boot = np_train[first_level_boot_indices[b], :]

        # Create the new dataset with IPCW weights
        df_train_boot = create_data_with_ipc_weights(
            data=pd.DataFrame(np_train_boot, columns=df_train_columns_name), params=data_generation_parameter
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
        _ ,pred = clf.predict_proba(data_generation_parameter['X_pred_point'].values)
        preds[b] = pred[0]

    return np.var(preds, ddof=1)

#########################################################################################################
' Data Generation functions '
def create_weibull_data(params: dict, random_state: int) -> pd.DataFrame:
    
    if params['X_pred_point'].shape[1] == 3:
        (shape_weibull, scale_weibull_base, rate_censoring, n, p_1, p_2, p_3) = (
            params["shape_weibull"],
            params["scale_weibull_base"],
            params["rate_censoring"],
            params["n"],
            params["p_1"], 
            params["p_2"],
            params["p_3"],)

        # Kovariaten
        rng = np.random.default_rng(random_state)
        X_1 = rng.normal(40, 5, n)
        X_2 = rng.binomial(1, 0.8, n)
        X_3 = rng.normal(80, 10, n)  


        # Lambda Weibull
        lambda_weibull = scale_weibull_base * np.exp(
            p_1 * X_1
            + p_2 * X_2  
            + p_3 * X_3 
        )

        # Ereigniszeiten und Zensierzeiten
        event_times = rng.weibull(shape_weibull, n) * lambda_weibull
        censoring_times = rng.exponential(1 / rate_censoring, n)
        observed_times = np.minimum(event_times, censoring_times)
        event_occurred = event_times <= censoring_times

        # Erstellung des Datensatzes
        data = pd.DataFrame(
            {
                "X_1": X_1,
                "X_2": X_2.astype(int),
                "X_3": X_3,
                "time": observed_times,
                "event": event_occurred.astype(int),
            }
        )

    elif params['X_pred_point'].shape[1] == 4:   
        (shape_weibull, scale_weibull_base, rate_censoring, n, p_1, p_2, p_3,p_4) = (
            params["shape_weibull"],
            params["scale_weibull_base"],
            params["rate_censoring"],
            params["n"],
            params["p_1"], 
            params["p_2"],
            params["p_3"],
            params["p_4"],)

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
    if data.shape[1]-2 == 3:
        X = data[["X_1", "X_2", "X_3"]]
    elif data.shape[1]-2 == 4:
        X = data[["X_1", "X_2", "X_3", "X_4"]]
        
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

def calculate_true_survival_probability(params):

    if params['X_pred_point'].shape[1] == 3:
        (
            shape_weibull,
            scale_weibull_base,
            p_1,
            p_2,
            p_3,
            individual,
        ) = (
            params["shape_weibull"],
            params["scale_weibull_base"],
            params["p_1"],
            params["p_2"],
            params["p_3"],
            params["X_pred_point"],
        )

        # Extrahieren der Kovariaten
        X_1 = individual['X_1'].values[0]
        X_2 = individual['X_2'].values[0]
        X_3 = individual['X_3'].values[0]

        # Berechnung des linearen Prädiktors (LP)
        LP =  X_1 * p_1 +X_2 * p_2 + X_3 * p_3

    elif params['X_pred_point'].shape[1] == 4:
        (
            shape_weibull,
            scale_weibull_base,
            p_1,
            p_2,
            p_3,
            p_4,
            individual,
        ) = (
            params["shape_weibull"],
            params["scale_weibull_base"],
            params["p_1"],
            params["p_2"],
            params["p_3"],
            params["p_4"],
            params["X_pred_point"],
        )

        # Extrahieren der Kovariaten
        X_1 = individual['X_1'].values[0]
        X_2 = individual['X_2'].values[0]
        X_3 = individual['X_3'].values[0]
        X_4 = individual['X_4'].values[0]

        # Berechnung des linearen Prädiktors (LP)
        LP =  X_1 * p_1 +X_2 * p_2 + X_3 * p_3 + X_4 * p_4

    # Individueller Skalenparameter lambda
    lambda_weibull = scale_weibull_base * np.exp(LP)

    # Überlebensfunktion
    S_t = np.exp(- (params["tau"] / lambda_weibull) ** shape_weibull)

    return S_t


#########################################################################################################
' Helper functions '
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


#########################################################################################################
' Simulation function '
def simulation(
    seed: int,
    data_generation_parameter: dict,
    params_rf: dict,
    train_models: bool ,
    ijk_wager_calc: bool,
    ijk_butt_calc: bool,
    ijk_jahn_calc: bool,
    jk_wager_calc: bool,
    boot_calc: bool):

    data = create_weibull_data(params=data_generation_parameter, random_state=seed)
    df_train, df_test = stratified_split(data=data, random_state=seed, test_size=0.3)
    df_train = create_data_with_ipc_weights(data=df_train, params=data_generation_parameter)
    df_test = create_data_with_ipc_weights(data=df_test, params=data_generation_parameter)

    if train_models:
        ### Weibull Modell ####
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
            aft.predict_survival_function(df=data_generation_parameter['X_pred_point'], times=data_generation_parameter["tau"]).iloc[0].tolist()
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
        _ ,rf_pred = clf.predict_proba(data_generation_parameter['X_pred_point'].values)

    else:
        wb_test_mse = 0.
        rf_test_mse = 0.0
        wb_pred = [0.0]
        rf_pred = [0.0]

    #### Variance Estimation ####
    ### Butt
    if ijk_butt_calc:
        ijk_butt_var, u1, u2 = calculate_ijk_butt_variance( clf=clf, X_pred_point=data_generation_parameter['X_pred_point'], df_train=df_train )
        ijk_u_butt_var = ijk_butt_var - u1
        ijk_u2_butt_var = ijk_butt_var - u2
        
    else:
        ijk_butt_var = 0.0
        ijk_u_butt_var = 0.0
        ijk_u2_butt_var = 0.0       

    ### Jahn
    if ijk_jahn_calc:
        ijk_jahn_var, u1, u2 =   calculate_ijk_jahn_variance(clf=clf, X_pred_point=data_generation_parameter['X_pred_point'], df_train=df_train)
        ijk_u_jahn_var = ijk_jahn_var - u1
        ijk_u2_jahn_var = ijk_jahn_var - u2
    else:
        ijk_jahn_var = 0.0
        ijk_u_jahn_var = 0.0
        ijk_u2_jahn_var = 0.0

    ### Wager
    if ijk_wager_calc:
        ijk_wager_var, u = calculate_ijk_wager_variance(clf=clf, X_pred_point=data_generation_parameter['X_pred_point'], df_train=df_train)
        ijk_u_wager_var = ijk_wager_var - u
    else:
        ijk_wager_var = 0.0
        ijk_u_wager_var = 0.0

    if jk_wager_calc:
        jk_wager_var, u = calculate_jk_variance(clf=clf, X_pred_point=data_generation_parameter['X_pred_point'], params_rf=params_rf, df_train=df_train)
        jk_u_wager_var = jk_wager_var - u
    else:
        jk_wager_var = 0.0
        jk_u_wager_var = 0.0

    ### boot
    if boot_calc[0]:
        boot_var = calculate_bootstrap_variance(params_rf=params_rf, df_train=df_train, seed=seed, B_1=boot_calc[1], data_generation_parameter=data_generation_parameter)
    else:
        boot_var = 0.0

    return (
        wb_test_mse,
        rf_test_mse,
        wb_pred,
        rf_pred,
        [ijk_butt_var, ijk_u_butt_var, ijk_u2_butt_var],
        [ijk_jahn_var, ijk_u_jahn_var, ijk_u2_jahn_var],
        [ijk_wager_var, ijk_u_wager_var],
        [jk_wager_var, jk_u_wager_var],
        boot_var,
        df_train['survived'].value_counts(normalize=True),
        df_test['survived'].value_counts(normalize=True),
    )


#########################################################################################################
' Save results '
def save_results(n,n_covariates, B_RF, boot_calc, seed, results1, results3, results5,
                 data_generation_parameter_1, data_generation_parameter_3, data_generation_parameter_5, params_rf):
    
    # create directory to save results
    n_sim = results1.shape[0]
    exp_name = f'(n_train){int(n*0.7)}__(B_RF){B_RF}__(B_1){boot_calc[1]}__(n_sim){n_sim}__(seed){seed}__{n_covariates}kovariates_test_boot'
    path = os.path.abspath('')
    if not os.path.exists(path + '/results/'+exp_name):
        os.makedirs(path + '/results/'+exp_name)

    _ = results1['portion_zero_weights_train'].mean() + results3['portion_zero_weights_train'].mean() +  results5['portion_zero_weights_train'].mean() + np.sum(data_generation_parameter_1['X_pred_point'].values[0])
    save_path = path + '/results/'+exp_name+'/'+str(_)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # save results
    results1.to_csv(save_path+f"/results1__(zero_weights){results1['portion_zero_weights_train'].mean().round(4)}__(seen_events){results1['portion_seen_events_train'].mean().round(4)}.csv")
    results3.to_csv(save_path+f"/results3__(zero_weights){results3['portion_zero_weights_train'].mean().round(4)}__(seen_events){results3['portion_seen_events_train'].mean().round(4)}.csv")
    results5.to_csv(save_path+f"/results5__(zero_weights){results5['portion_zero_weights_train'].mean().round(4)}__(seen_events){results5['portion_seen_events_train'].mean().round(4)}.csv")
    
    # save json file with the parameters
    with open(save_path + '/setting.json', 'w') as file:
        json.dump({'n_covariates': n_covariates,'n_sim':n_sim, 'n_train': int(n*0.7), 'n': n, 'B_RF': B_RF, 'B_1': boot_calc[1], 'seed': seed, 
                   'X_pred_point': str(data_generation_parameter_1['X_pred_point'].values[0]),
                    'shape_weibull': data_generation_parameter_1['shape_weibull'],
                   'data_generation_parameter_0_1': str(data_generation_parameter_1),
                   'data_generation_parameter_0_3': str(data_generation_parameter_3),
                   'data_generation_parameter_0_5': str(data_generation_parameter_5),
                   'params_rf': str(params_rf),
                   'portion_zero_weights_train[1,3,5]':[  results1['portion_zero_weights_train'].mean().round(4),results3['portion_zero_weights_train'].mean().round(4),results5['portion_zero_weights_train'].mean().round(4)] ,
                   'portion_seen_events_train[1,3,5]': [results1['portion_seen_events_train'].mean().round(4),results3['portion_seen_events_train'].mean().round(4),results5['portion_seen_events_train'].mean().round(4)],
                   'true_survival_probability[1,3,5]': [calculate_true_survival_probability(data_generation_parameter_1),
                                                        calculate_true_survival_probability(data_generation_parameter_3),
                                                        calculate_true_survival_probability(data_generation_parameter_5)],
                    'wb_test_mse_ipcw[1,3,5]': [results1['wb_test_mse'].mean(), 
                                                results3['wb_test_mse'].mean(), 
                                                results5['wb_test_mse'].mean()],
                    'rf_test_mse_ipcw[1,3,5]': [results1['rf_test_mse'].mean(),
                                                results3['rf_test_mse'].mean(),
                                                results5['rf_test_mse'].mean()],
                   }, file)
        
    return save_path


#########################################################################################################
'Plot functions'

def plot_pred_bias(exp_save_path, y1=None,y2=None):
    with open(exp_save_path + '/setting.json') as f:
        exp_settings = json.load(f)
    S_t = exp_settings["true_survival_probability[1,3,5]"]

    # lade results
    results1 = pd.read_csv(exp_save_path + f"/results1__(zero_weights){exp_settings['portion_zero_weights_train[1,3,5]'][0]}__(seen_events){exp_settings['portion_seen_events_train[1,3,5]'][0]}.csv")
    results3 = pd.read_csv(exp_save_path + f"/results3__(zero_weights){exp_settings['portion_zero_weights_train[1,3,5]'][1]}__(seen_events){exp_settings['portion_seen_events_train[1,3,5]'][1]}.csv")
    results5 = pd.read_csv(exp_save_path + f"/results5__(zero_weights){exp_settings['portion_zero_weights_train[1,3,5]'][2]}__(seen_events){exp_settings['portion_seen_events_train[1,3,5]'][2]}.csv")

    pred_bias_1 = [ (results1['wb_pred'].mean()-S_t[0])/S_t[0] * 100  , (results1['rf_pred'].mean()-S_t[0])/S_t[0] * 100]
    pred_bias_3 = [ (results3['wb_pred'].mean()-S_t[1])/S_t[1] * 100  , (results3['rf_pred'].mean()-S_t[1])/S_t[1] * 100]
    pred_bias_5 = [ (results5['wb_pred'].mean()-S_t[2])/S_t[2] * 100  , (results5['rf_pred'].mean()-S_t[2])/S_t[2] * 100]
    pred_bias = [ pred_bias_1,   pred_bias_3,   pred_bias_5  ]

    pred_error_1 = [np.std((results1['wb_pred']-S_t[0])/S_t[0]* 100,ddof=1) , np.std((results1['rf_pred']-S_t[0])/S_t[0]* 100,ddof=1)]
    pred_error_3 = [np.std((results3['wb_pred']-S_t[1])/S_t[1]* 100,ddof=1) , np.std((results3['rf_pred']-S_t[1])/S_t[1]* 100,ddof=1)]
    pred_error_5 = [np.std((results5['wb_pred']-S_t[2])/S_t[2]* 100,ddof=1) , np.std((results5['rf_pred']-S_t[2])/S_t[2]* 100,ddof=1)]
    bias_errors =  [ pred_error_1,  pred_error_3,  pred_error_5 ]

    # Plot Prediction Bias
    names = ['Weibull AFT', 'DCTB']
    weights_zero = [ round(_, 2) for _ in exp_settings['portion_zero_weights_train[1,3,5]']]

    plt.figure(figsize=(10, 5))
    colors = ['green', 'magenta'] 
    offset = np.linspace(-0.01, 0.01, len(names)) 

    for j, name in enumerate(names):
        for i, weight in enumerate(weights_zero):
            plt.errorbar(weight + offset[j], pred_bias[i][j], yerr=bias_errors[i][j]*1.96, fmt='o', color=colors[j], label=name if i == 0 else "")

    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel(r'$\frac{\mid \{w_i=0\} \mid}{\mid \{w_i\} \mid}$')
    plt.ylabel(r'rel. bias of $\hat{S}(\tau\mid X_{pred})$'+str(' (in %)'))
    plt.title(r'Bias of predicted survival propability at $\tau$ for $X_{pred}= $'+str(exp_settings['X_pred_point'] ))
    plt.grid(True)
    plt.xticks(weights_zero)

    plt.legend(title='Model', loc='upper left', fancybox=True, shadow=True, ncol=2, bbox_to_anchor=(0.35, -0.2))

    a = 0.7
    b= -0.15
    plt.text(-0.12, -0.25, f"(dots are mean bias,\n  errorbars are based on 1.96 * empirical std.)\n\n Simulation study params: \n n_sim={exp_settings['n_sim']}, n_train = {exp_settings['n_train'] }, B ={exp_settings['B_RF']}, B_1={exp_settings['B_1'] }\n ", 
            ha='left', va='center', transform=plt.gca().transAxes)
    plt.text(a,b, r"$S(\tau\mid X_{pred})$  (in %)  ="+ f" {[ round(_, 2) for _ in S_t]}" ,
            ha='left', va='center', transform=plt.gca().transAxes)
    plt.text(a,b-0.1, f"Seen events (in %) = {[round(_,2) for _ in exp_settings['portion_seen_events_train[1,3,5]']]} ",
            ha='left', va='center', transform=plt.gca().transAxes)
    plt.text(a,b-0.2, r"....... for each setting with $\frac{\mid \{w_i=0\} \mid}{\mid \{w_i\} \mid}$", 
            ha='left', va='center', transform=plt.gca().transAxes) 
    if y1 is not None and y2 is not None:
        plt.ylim(y1,y2)
        plt.savefig(exp_save_path + f'/pred_S_bias(n_train){exp_settings["n_train"]}__(B_RF){exp_settings["B_RF"]}__(n_sim){exp_settings["n_sim"]}__covariates{exp_settings["n_covariates"]}__{y1+y2}.png', bbox_inches='tight')
    else:
        plt.savefig(exp_save_path + f'/pred_S_bias(n_train){exp_settings["n_train"]}__(B_RF){exp_settings["B_RF"]}__(n_sim){exp_settings["n_sim"]}__covariates{exp_settings["n_covariates"]}.png', bbox_inches='tight')

def plot_var_bias_without_u2(exp_save_path, y1=None, y2=None,patient=''):

    with open(exp_save_path + '/setting.json') as f:
        exp_settings = json.load(f)

    # lade results
    results1 = pd.read_csv(exp_save_path + f"/results1__(zero_weights){exp_settings['portion_zero_weights_train[1,3,5]'][0]}__(seen_events){exp_settings['portion_seen_events_train[1,3,5]'][0]}.csv")
    results3 = pd.read_csv(exp_save_path + f"/results3__(zero_weights){exp_settings['portion_zero_weights_train[1,3,5]'][1]}__(seen_events){exp_settings['portion_seen_events_train[1,3,5]'][1]}.csv")
    results5 = pd.read_csv(exp_save_path + f"/results5__(zero_weights){exp_settings['portion_zero_weights_train[1,3,5]'][2]}__(seen_events){exp_settings['portion_seen_events_train[1,3,5]'][2]}.csv")

    # True Standard Deviations
    true_std_1 = results1['rf_pred'].std(ddof=1)
    true_std_3 = results3['rf_pred'].std(ddof=1)
    true_std_5 = results5['rf_pred'].std(ddof=1)

    # Bias and Error of the Standard Deviation Estimates (ohne u2 Schätzer)
    exclude_keys = ['ijk_butt_var',  'ijk_u_butt_var',  'ijk_u2_butt_var',
                    'ijk_jahn_var',  'ijk_u2_jahn_var', 'jk_wager_var',
                    ]
    columns = [col for col in results1.columns[9:] if not any(key in col for key in exclude_keys)]
    columns.remove('jk_u_wager_var')

    def calc_bias_and_error(results, true_std, columns):
        bias = [(np.mean(results[col].apply(lambda x: np.sqrt(x) if x >= 0 else 0)) - true_std) / true_std * 100 for col in columns]
        error = [np.std((results[col].apply(lambda x: np.sqrt(x) if x >= 0 else 0) - true_std) / true_std * 100, ddof=1) for col in columns]
        return bias, error

    var_bias_1, var_error_1 = calc_bias_and_error(results1, true_std_1, columns)
    var_bias_3, var_error_3 = calc_bias_and_error(results3, true_std_3, columns)
    var_bias_5, var_error_5 = calc_bias_and_error(results5, true_std_5, columns)

    var_bias = [var_bias_1, var_bias_3, var_bias_5]
    errors = [var_error_1, var_error_3, var_error_5]

    names = [col[:-4] for col in columns]  # names of the methods
    names = [r'$\hat{V}_{IJ-U}^{wB}$',r'$\hat{V}_{IJ-U}^{B}$',r'$\hat{V}_{Boot}$']
    weights_zero = [round(_, 2) for _ in exp_settings['portion_zero_weights_train[1,3,5]']]

    # Erzeuge ein Plot
    plt.figure(figsize=(10, 5))
    colors = [
        (0.0, 0.25, 0.74),  # Blau
        (0.85, 0.33, 0.1),  # Orange
        (0.47, 0.67, 0.19)  # Grün
        ]

    # Plotten der Punkte mit Fehlerbalken und Legende
    offset = np.linspace(-0.03, 0.03, len(names))  # Kleinere Versatzwerte für die x-Werte
    for j, name in enumerate(names):
        for i, weight in enumerate(weights_zero):
            plt.errorbar(weight + offset[j], var_bias[i][j], yerr=errors[i][j], fmt='o', color=colors[j], label=name if i == 0 else "")

    # Achsenbeschriftungen und Titel hinzufügen
    plt.axhline(y=0, color='red', linestyle='--')
    plt.grid(True)
    plt.xticks(weights_zero)
    plt.xlabel(r'$p_{w_0}$')
    plt.ylabel("RB (in %)")
    plt.legend(title='Estimator', loc='upper left')
    plt.ylim(y1, y2)

    plt.savefig(exp_save_path + f'/pred_var_bias(n_train){exp_settings["n_train"]}__(B_RF){exp_settings["B_RF"]}__(n_sim){exp_settings["n_sim"]}__covariates{exp_settings["n_covariates"]}_{patient}.jpeg', bbox_inches='tight',
                dpi = 300)

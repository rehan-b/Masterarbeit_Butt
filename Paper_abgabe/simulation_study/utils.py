import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter, WeibullAFTFitter
from sksurv.util import Surv
from sklearn.model_selection import train_test_split
from class_DecisionTreeBaggingClassifier import DecisionTreeBaggingClassifier
import os, json
import matplotlib.pyplot as plt
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

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
    bias_correction2 = (1/n_plus**2)  * np.var(T_N_b, axis=0, ddof=1)* n / B * np.sum( ( 1 / (weights[weights > 0] ) )) 

    return biased_var_estimate , bias_correction[0], bias_correction2[0]

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

    #plt.savefig(exp_save_path + f'\\pred_var_bias(n_train){exp_settings["n_train"]}__(B_RF){exp_settings["B_RF"]}__(n_sim){exp_settings["n_sim"]}__covariates{exp_settings["n_covariates"]}_{patient}.jpeg', bbox_inches='tight',
    #            dpi = 300)


#########################################################################################################
# Optimized end-to-end pipeline (Boot + IJK-Jahn + IJK-Wager only)
REQUIRED_VARIANCE_COLUMNS = [
    "ijk_jahn_var",
    "ijk_u_jahn_var",
    "ijk_wager_var",
    "ijk_u_wager_var",
    "boot_var",
]


def calculate_ijk_jahn_variance_from_arrays(
    T_N_b: np.ndarray, pred: np.ndarray, N_bi: np.ndarray, weights: np.ndarray
):
    """Fast IJK-Jahn variance from precomputed tree predictions."""
    B, n = N_bi.shape
    n_plus = np.sum(weights > 0)

    cov_i = ((N_bi - n * weights.reshape(1, -1)).T @ (T_N_b - pred)) / B
    cov_i_hoch2 = cov_i**2
    array = cov_i_hoch2 / ((weights.reshape(-1, 1)) ** 2)

    biased_var_estimate = np.sum(array[~np.isnan(array) & ~np.isinf(array)], axis=0) * (1 / n_plus**2)
    bias_correction = (1 / n_plus**2) * np.var(T_N_b, axis=0, ddof=1) * n / B * np.sum((1 / (weights[weights > 0])) - 1)
    bias_correction2 = (1 / n_plus**2) * np.var(T_N_b, axis=0, ddof=1) * n / B * np.sum((1 / (weights[weights > 0])))

    return biased_var_estimate, bias_correction[0], bias_correction2[0]


def calculate_ijk_wager_variance_from_arrays(
    T_N_b: np.ndarray, pred: np.ndarray, N_bi: np.ndarray
):
    """Fast IJK-Wager variance from precomputed tree predictions."""
    B, n = N_bi.shape

    cov_i = ((N_bi - 1).T @ (T_N_b - pred)) / B
    cov_i_hoch2 = cov_i**2

    biased_var_estimate = np.sum(cov_i_hoch2)
    bias_correction = n / B * np.var(T_N_b, axis=0, ddof=1)

    return biased_var_estimate, bias_correction[0]


def simulation_core(
    seed: int,
    data_generation_parameter: dict,
    params_rf: dict,
    train_models: bool,
    boot_B1: int,
):
    """Single simulation run with only required variance estimators."""
    data = create_weibull_data(params=data_generation_parameter, random_state=seed)
    df_train, df_test = stratified_split(data=data, random_state=seed, test_size=0.3)
    df_train = create_data_with_ipc_weights(data=df_train, params=data_generation_parameter)
    df_test = create_data_with_ipc_weights(data=df_test, params=data_generation_parameter)

    if not train_models:
        return {
            "wb_test_mse": 0.0,
            "rf_test_mse": 0.0,
            "wb_pred": 0.0,
            "rf_pred": 0.0,
            "ijk_jahn_var": 0.0,
            "ijk_u_jahn_var": 0.0,
            "ijk_wager_var": 0.0,
            "ijk_u_wager_var": 0.0,
            "boot_var": 0.0,
            "portion_zero_weights_train": 0.0,
            "portion_seen_events_train": 0.0,
            "portion_zero_weights_test": 0.0,
            "portion_seen_events_test": 0.0,
        }

    aft = WeibullAFTFitter()
    aft.fit(
        df=df_train.drop(["weights_ipcw", "survived"], axis=1),
        duration_col="time",
        event_col="event",
    )

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
    wb_pred = (
        aft.predict_survival_function(
            df=data_generation_parameter["X_pred_point"],
            times=data_generation_parameter["tau"],
        )
        .iloc[0]
        .tolist()[0]
    )

    params_rf_local = dict(params_rf)
    params_rf_local["random_state"] = seed
    clf = DecisionTreeBaggingClassifier(params_rf_local)
    clf.fit(
        X=df_train.drop(["time", "event", "weights_ipcw", "survived"], axis=1).values,
        y=df_train["survived"].values,
        sample_weights=df_train["weights_ipcw"].values,
    )

    _, pred_test = clf.predict_proba(
        df_test.drop(["weights_ipcw", "survived", "time", "event"], axis=1).values
    )
    rf_test_mse = ipc_weighted_mse(
        y_true=df_test["survived"].values,
        y_pred=pred_test,
        sample_weight=df_test["weights_ipcw"].values,
    )

    T_N_b, pred_point = clf.predict_proba(data_generation_parameter["X_pred_point"].values)
    rf_pred = pred_point[0]
    N_bi = clf.nbi
    weights = df_train["weights_ipcw"].values

    ijk_jahn_var, jahn_u1, _ = calculate_ijk_jahn_variance_from_arrays(
        T_N_b=T_N_b,
        pred=pred_point,
        N_bi=N_bi,
        weights=weights,
    )
    ijk_wager_var, wager_u = calculate_ijk_wager_variance_from_arrays(
        T_N_b=T_N_b,
        pred=pred_point,
        N_bi=N_bi,
    )

    boot_var = calculate_bootstrap_variance(
        params_rf=params_rf_local,
        df_train=df_train,
        seed=seed,
        B_1=boot_B1,
        data_generation_parameter=data_generation_parameter,
    )

    return {
        "wb_test_mse": wb_test_mse,
        "rf_test_mse": rf_test_mse,
        "wb_pred": wb_pred,
        "rf_pred": rf_pred,
        "ijk_jahn_var": ijk_jahn_var,
        "ijk_u_jahn_var": ijk_jahn_var - jahn_u1,
        "ijk_wager_var": ijk_wager_var,
        "ijk_u_wager_var": ijk_wager_var - wager_u,
        "boot_var": boot_var,
        "portion_zero_weights_train": df_train["weights_ipcw"].eq(0).mean(),
        "portion_seen_events_train": (df_train["survived"] == 0).mean(),
        "portion_zero_weights_test": df_test["weights_ipcw"].eq(0).mean(),
        "portion_seen_events_test": (df_test["survived"] == 0).mean(),
    }




def _simulation_task(task):
    """Top-level worker helper for multiprocessing."""
    key, sim_index, seed, data_params, params_rf, b1 = task
    result = simulation_core(
        seed=seed + sim_index,
        data_generation_parameter=data_params,
        params_rf=params_rf,
        train_models=True,
        boot_B1=b1,
    )
    return key, result


def _print_progress(done: int, total: int, bar_len: int = 32):
    """Print a simple in-place console progress bar."""
    if total <= 0:
        return
    frac = done / total
    filled = int(bar_len * frac)
    bar = "#" * filled + "-" * (bar_len - filled)
    print(f"\rSimulation progress: [{bar}] {done}/{total} ({frac * 100:5.1f}%)", end="", flush=True)
    if done == total:
        print()

def run_simulation_and_save(
    n_sim: int,
    seed: int,
    n_covariates: int,
    n: int,
    B_RF: int,
    B_1: int,
    data_generation_parameter_1: dict,
    data_generation_parameter_3: dict,
    data_generation_parameter_5: dict,
    params_rf: dict,
    n_jobs: int = -1,
):
    """Run all three zero-weight scenarios and save results in one folder."""
    scenario_map = {
        "1": data_generation_parameter_1,
        "3": data_generation_parameter_3,
        "5": data_generation_parameter_5,
    }

    if n_jobs in (None, -1):
        max_workers = os.cpu_count() or 1
    else:
        max_workers = max(1, int(n_jobs))

    tasks = []
    for key, data_params in scenario_map.items():
        for i in range(n_sim):
            tasks.append((key, i, seed, data_params, params_rf, B_1))

    results_rows = {"1": [], "3": [], "5": []}
    total_tasks = len(tasks)
    done_tasks = 0
    _print_progress(done_tasks, total_tasks)

    if max_workers == 1:
        for task in tasks:
            key, row = _simulation_task(task)
            results_rows[key].append(row)
            done_tasks += 1
            _print_progress(done_tasks, total_tasks)
    else:
        chunksize = max(1, len(tasks) // (max_workers * 4))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_simulation_task, task) for task in tasks]
            for future in as_completed(futures):
                key, row = future.result()
                results_rows[key].append(row)
                done_tasks += 1
                _print_progress(done_tasks, total_tasks)

    results = {key: pd.DataFrame(rows) for key, rows in results_rows.items()}

    exp_name = f"(n_train){int(n*0.7)}__(B_RF){B_RF}__(B_1){B_1}__(n_sim){n_sim}__(seed){seed}__{n_covariates}kovariates"
    base_dir = os.path.join(os.path.abspath(""), "results", exp_name)
    os.makedirs(base_dir, exist_ok=True)

    suffix = (
        results["1"]["portion_zero_weights_train"].mean()
        + results["3"]["portion_zero_weights_train"].mean()
        + results["5"]["portion_zero_weights_train"].mean()
        + np.sum(data_generation_parameter_1["X_pred_point"].values[0])
    )
    save_path = os.path.join(base_dir, str(suffix))
    os.makedirs(save_path, exist_ok=True)

    for key in ["1", "3", "5"]:
        df = results[key]
        df.to_csv(
            os.path.join(
                save_path,
                f"results{key}__(zero_weights){df['portion_zero_weights_train'].mean().round(4)}__(seen_events){df['portion_seen_events_train'].mean().round(4)}.csv",
            ),
            index=False,
        )

    with open(os.path.join(save_path, "setting.json"), "w") as file:
        json.dump(
            {
                "n_covariates": n_covariates,
                "n_sim": n_sim,
                "n_train": int(n * 0.7),
                "n": n,
                "B_RF": B_RF,
                "B_1": B_1,
                "seed": seed,
                "X_pred_point": str(data_generation_parameter_1["X_pred_point"].values[0]),
                "shape_weibull": data_generation_parameter_1["shape_weibull"],
                "data_generation_parameter_0_1": str(data_generation_parameter_1),
                "data_generation_parameter_0_3": str(data_generation_parameter_3),
                "data_generation_parameter_0_5": str(data_generation_parameter_5),
                "params_rf": str(params_rf),
                "portion_zero_weights_train[1,3,5]": [
                    results["1"]["portion_zero_weights_train"].mean().round(4),
                    results["3"]["portion_zero_weights_train"].mean().round(4),
                    results["5"]["portion_zero_weights_train"].mean().round(4),
                ],
                "portion_seen_events_train[1,3,5]": [
                    results["1"]["portion_seen_events_train"].mean().round(4),
                    results["3"]["portion_seen_events_train"].mean().round(4),
                    results["5"]["portion_seen_events_train"].mean().round(4),
                ],
                "true_survival_probability[1,3,5]": [
                    calculate_true_survival_probability(data_generation_parameter_1),
                    calculate_true_survival_probability(data_generation_parameter_3),
                    calculate_true_survival_probability(data_generation_parameter_5),
                ],
                "wb_test_mse_ipcw[1,3,5]": [
                    results["1"]["wb_test_mse"].mean(),
                    results["3"]["wb_test_mse"].mean(),
                    results["5"]["wb_test_mse"].mean(),
                ],
                "rf_test_mse_ipcw[1,3,5]": [
                    results["1"]["rf_test_mse"].mean(),
                    results["3"]["rf_test_mse"].mean(),
                    results["5"]["rf_test_mse"].mean(),
                ],
            },
            file,
        )

    return save_path


def _load_results_by_setting(exp_save_path: str):
    with open(exp_save_path + '/setting.json') as f:
        exp_settings = json.load(f)

    results1 = pd.read_csv(exp_save_path + f"/results1__(zero_weights){exp_settings['portion_zero_weights_train[1,3,5]'][0]}__(seen_events){exp_settings['portion_seen_events_train[1,3,5]'][0]}.csv")
    results3 = pd.read_csv(exp_save_path + f"/results3__(zero_weights){exp_settings['portion_zero_weights_train[1,3,5]'][1]}__(seen_events){exp_settings['portion_seen_events_train[1,3,5]'][1]}.csv")
    results5 = pd.read_csv(exp_save_path + f"/results5__(zero_weights){exp_settings['portion_zero_weights_train[1,3,5]'][2]}__(seen_events){exp_settings['portion_seen_events_train[1,3,5]'][2]}.csv")
    return exp_settings, results1, results3, results5




def create_plots_from_notebooks(
    exp_save_path: str,
    patient: str = "averageS",
    corr_xlims=None,
    strip_xlim=None,
    rb_xlim=None,
):
    """Execute plot notebooks in-process and pass optional x-axis limits via environment."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    notebooks = [
        "corr_plots.ipynb",
        "RB_est_var.ipynb",
        "stripplot_and_CP.ipynb",
    ]

    target_path = os.path.abspath(exp_save_path)
    os.environ["EXP_SAVE_PATH"] = target_path
    os.environ["PATIENT_LABEL"] = patient

    if corr_xlims is not None:
        if len(corr_xlims) != 3:
            raise ValueError("corr_xlims must contain exactly 3 (min,max) tuples")
        os.environ["CORR_XLIM_1"] = f"{float(corr_xlims[0][0])},{float(corr_xlims[0][1])}"
        os.environ["CORR_XLIM_3"] = f"{float(corr_xlims[1][0])},{float(corr_xlims[1][1])}"
        os.environ["CORR_XLIM_5"] = f"{float(corr_xlims[2][0])},{float(corr_xlims[2][1])}"

    if strip_xlim is not None:
        os.environ["STRIP_XLIM"] = f"{float(strip_xlim[0])},{float(strip_xlim[1])}"

    if rb_xlim is not None:
        os.environ["RB_XLIM"] = f"{float(rb_xlim[0])},{float(rb_xlim[1])}"

    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    if existing_pythonpath:
        os.environ["PYTHONPATH"] = base_dir + os.pathsep + existing_pythonpath
    else:
        os.environ["PYTHONPATH"] = base_dir

    if os.name == "nt":
        try:
            import asyncio
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        except Exception:
            pass

    try:
        import nbformat
        from nbclient import NotebookClient
    except ImportError as exc:
        raise RuntimeError(
            "Notebook execution requires Python packages 'nbformat' and 'nbclient'. "
            "Install them in your environment (e.g. pip install nbformat nbclient)."
        ) from exc

    for nb in notebooks:
        nb_path = os.path.join(base_dir, nb)
        with open(nb_path, "r", encoding="utf-8") as f:
            notebook = nbformat.read(f, as_version=4)

        client = NotebookClient(notebook, timeout=1200, kernel_name="python3")
        client.execute(cwd=target_path)

        with open(nb_path, "w", encoding="utf-8") as f:
            nbformat.write(notebook, f)

    return {"notebooks_executed": notebooks, "exp_save_path": target_path}

def create_all_plots(exp_save_path: str):
    """Generate all core + additional plots for one experiment folder."""
    exp_settings, results1, results3, results5 = _load_results_by_setting(exp_save_path)
    S_t = exp_settings["true_survival_probability[1,3,5]"]
    results_map = {"0.1": results1, "0.3": results3, "0.5": results5}

    # 1) Prediction bias plot
    plot_pred_bias(exp_save_path)

    # 2) Variance RB plot (only required estimators)
    methods = ["ijk_u_jahn_var", "ijk_u_wager_var", "boot_var"]
    labels = [r'$\hat{V}_{IJ-U}^{wB}$', r'$\hat{V}_{IJ-U}^{B}$', r'$\hat{V}_{Boot}$']
    weights_zero = [round(_, 2) for _ in exp_settings['portion_zero_weights_train[1,3,5]']]

    plt.figure(figsize=(10, 5))
    offsets = np.linspace(-0.02, 0.02, len(methods))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for j, (m, label) in enumerate(zip(methods, labels)):
        for i, wz in enumerate(weights_zero):
            df = list(results_map.values())[i]
            true_var = df['rf_pred'].var(ddof=1)
            est_std = np.sqrt(np.maximum(df[m], 0))
            true_std = np.sqrt(true_var)
            rb = (est_std.mean() - true_std) / true_std * 100
            err = ((est_std - true_std) / true_std * 100).std(ddof=1)
            plt.errorbar(wz + offsets[j], rb, yerr=1.96 * err, fmt='o', color=colors[j], label=label if i == 0 else "")

    plt.axhline(0, color='red', linestyle='--')
    plt.grid(True)
    plt.xticks(weights_zero)
    plt.xlabel(r'$p_{w_0}$')
    plt.ylabel('rel. bias of std-estimator (in %)')
    plt.legend(title='Estimator')
    plt.tight_layout()
    plt.savefig(os.path.join(exp_save_path, 'variance_rb_required_estimators.png'), dpi=300)
    plt.close()

    # 3) NEW: Distribution plot of RF prediction by scenario
    plt.figure(figsize=(10, 5))
    data = [results1['rf_pred'].values, results3['rf_pred'].values, results5['rf_pred'].values]
    plt.boxplot(data, labels=[f"p_w0={w}" for w in weights_zero], showfliers=False)
    for i, s in enumerate(S_t, start=1):
        plt.scatter(i, s, color='red', marker='x', s=80, label='True S(tau|X_pred)' if i == 1 else "")
    plt.ylabel('RF prediction at tau')
    plt.xlabel('Scenario')
    plt.title('RF prediction distribution across scenarios')
    plt.grid(True, axis='y', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(exp_save_path, 'rf_prediction_distribution.png'), dpi=300)
    plt.close()

    # 4) NEW: Estimator mean vs empirical variance plot
    plt.figure(figsize=(10, 5))
    emp_vars = [results1['rf_pred'].var(ddof=1), results3['rf_pred'].var(ddof=1), results5['rf_pred'].var(ddof=1)]
    x = np.arange(len(weights_zero))
    width = 0.2
    plt.bar(x - 1.5 * width, emp_vars, width=width, label='empirical var(rf_pred)')
    plt.bar(x - 0.5 * width, [results1['ijk_u_jahn_var'].mean(), results3['ijk_u_jahn_var'].mean(), results5['ijk_u_jahn_var'].mean()], width=width, label='mean ijk_u_jahn_var')
    plt.bar(x + 0.5 * width, [results1['ijk_u_wager_var'].mean(), results3['ijk_u_wager_var'].mean(), results5['ijk_u_wager_var'].mean()], width=width, label='mean ijk_u_wager_var')
    plt.bar(x + 1.5 * width, [results1['boot_var'].mean(), results3['boot_var'].mean(), results5['boot_var'].mean()], width=width, label='mean boot_var')
    plt.xticks(x, [f"p_w0={w}" for w in weights_zero])
    plt.ylabel('Variance')
    plt.title('Variance estimator means vs empirical prediction variance')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(exp_save_path, 'variance_estimator_comparison.png'), dpi=300)
    plt.close()

    return {
        "plots": [
            "pred_S_bias...png",
            "variance_rb_required_estimators.png",
            "rf_prediction_distribution.png",
            "variance_estimator_comparison.png",
        ]
    }

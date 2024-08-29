import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from sklearn.ensemble import RandomForestClassifier
from lifelines import KaplanMeierFitter, WeibullAFTFitter
from sksurv.metrics import concordance_index_ipcw
from sksurv.util import Surv
from sklearn.model_selection import train_test_split



def create_new_dataset_with_ipcw_weights(
    data: pd.DataFrame, t: np.float64, kmf: KaplanMeierFitter
) -> pd.DataFrame:
    new_data = data.copy()

    new_data.loc[(data["time"] <= t) & (data["event"] == 1), "survived"] = int(0)
    new_data.loc[(data["time"] >= t) & (data["event"] == 0), "survived"] = int(1)
    new_data.loc[(data["time"] >= t) & (data["event"] == 1), "survived"] = int(1)
    new_data.loc[(data["time"] <= t) & (data["event"] == 0), "survived"] = int(999)

    new_data["survived"] = new_data["survived"].astype(int)

    ipcw_weights = 1 / kmf.survival_function_at_times(new_data["time"])
    ipcw_weight_tau = 1 / kmf.survival_function_at_times(t)
    new_data["weights_ipcw"] = np.where(
        new_data["survived"] == 1,
        ipcw_weight_tau,
        np.where(new_data["survived"] == 0, ipcw_weights, 0),
    )
    new_data["weights_ipcw"] = new_data["weights_ipcw"] / new_data["weights_ipcw"].sum()

    return pd.DataFrame(new_data)


def train_test_split_into_df(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    # Reset index of train and test DataFrames
    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)

    # Convert y_train and y_test to DataFrames
    y_train_df = pd.DataFrame(y_train, columns=["event", "time"])
    y_test_df = pd.DataFrame(y_test, columns=["event", "time"])

    # Assign event and time columns to train and test DataFrames
    df_train[["event", "time"]] = y_train_df[["event", "time"]]
    df_test[["event", "time"]] = y_test_df[["event", "time"]]

    return df_train, df_test


def create_train_test_data(params: dict) -> pd.DataFrame:

    ### Parameter für Weibull-Verteilung und Censoring ###
    shape_weibull = params.get('shape_weibull')
    scale_weibull_base = params.get('scale_weibull_base')
    rate_censoring = params.get('rate_censoring')
    n = params.get('n')
    b_bloodp = params.get('b_bloodp')
    b_diab = params.get('b_diab')
    b_age = params.get('b_age')
    b_bmi = params.get('b_bmi')
    b_kreat = params.get('b_kreat')
    seed = params.get('seed')
    tau = params.get('tau')

    ### Generierung der Kovariaten ###
    rng = np.random.default_rng(seed)
    bmi = rng.normal(25, 5, n)
    blood_pressure = rng.binomial(1, 0.3, n)
    kreatinkinase = rng.lognormal(mean=5, sigma=1, size=n)
    kreatinkinase = np.clip(kreatinkinase, 30, 8000)
    diabetes = rng.binomial(1, 0.2, n)
    age = rng.normal(50, 10, n)  #

    ### Weibull-Verteilung ###
    lambda_weibull = scale_weibull_base * np.exp(
        b_bloodp * blood_pressure
        + b_diab * diabetes  # Linearer Einfluss von hohem Blutdruck
        + b_age * age  # Linearer Einfluss von Diabetes
        + b_bmi * (bmi - 25) ** 2  # Linearer Einfluss des Alters
        + b_kreat  # Quadratischer Einfluss des BMI
        * np.log(kreatinkinase)  # Exponentieller Einfluss der Kreatinkinase
    )

    ### Generierung der Ereigniszeiten/Zensierzeiten basierend auf der Weibull-/ZensierVerteilung
    event_times = rng.weibull(shape_weibull, n) * lambda_weibull
    censoring_times = rng.exponential(1 / rate_censoring, n)
    observed_times = np.minimum(event_times, censoring_times)
    event_occurred = event_times <= censoring_times

    ### Erstellung des Datensatzes ohne die Transformationen ###
    data = pd.DataFrame(
        {
            "bmi": bmi,
            "blood_pressure": blood_pressure.astype(int),
            "kreatinkinase": kreatinkinase,
            "diabetes": diabetes.astype(int),
            "age": age,
            "t": observed_times,
            "event": event_occurred.astype(int),
        }
    )
    #print("Data shape:", data.shape)
    #print(f'{(data["event"] ==1).sum()/n  * 100} % of the data has an event')


    ### Startified Split ###
    X = data[['bmi', 'blood_pressure', 'kreatinkinase', 'diabetes', 'age']]
    y = Surv.from_arrays(event=data['event'], time=data['t'])
    df_train, df_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
    df_train, df_test = train_test_split_into_df(df_train=df_train, df_test=df_test, y_train=y_train, y_test=y_test)


    ### cut data at tau // ipcw weights ###
    kmf = KaplanMeierFitter()
    kmf.fit(df_train['time'], event_observed=1-df_train['event'])
    df_train = create_new_dataset_with_ipcw_weights(data=df_train,t=tau, kmf=kmf)
    df_test = create_new_dataset_with_ipcw_weights(data=df_test,t=tau, kmf=kmf)

    ### calculate portion of events and censored data after cut  for training data ###
    portions_at_cutpoint = df_train['survived'].value_counts(normalize=True)
    if 999 in portions_at_cutpoint.keys():
        portion_censored_after_cut_train = portions_at_cutpoint[999]
    else:
        portion_censored_after_cut_train = 0

    n_events_after_cut_train = portions_at_cutpoint[1] * df_train.shape[0] 

    ### calculate portion of events and censored data after cut  for test data ###
    portions_at_cutpoint_test = df_test['survived'].value_counts(normalize=True)

    if 999 in portions_at_cutpoint_test.keys():
        portion_censored_after_cut_test = portions_at_cutpoint_test[999]
    else:
        portion_censored_after_cut_test = 0

    n_events_after_cut_test = portions_at_cutpoint_test[1] * df_test.shape[0]

    return df_train, df_test, n_events_after_cut_train, portion_censored_after_cut_train, n_events_after_cut_test, portion_censored_after_cut_test


def ipc_weighted_mse(y_true, y_pred, sample_weight):
    return np.average((y_true - y_pred) ** 2, weights=sample_weight)


def get_Nbi(lists):
    arr = np.array(lists)
    max_value = arr.max()
    counts = np.apply_along_axis(
        lambda x: np.bincount(x, minlength=max_value + 1), axis=1, arr=arr
    )
    return counts


def inf_JK_bagged_variance(
    N_bi: np.ndarray, T_N_b: np.ndarray, weights: np.ndarray = None
):
    B, n = N_bi.shape
    T_N_b_mean = np.mean(T_N_b, axis=0)
    m = np.count_nonzero(weights)

    cov_i = ((N_bi - n * weights[0]).T @ (T_N_b - T_N_b_mean)) / B
    cov_i_hoch2 = cov_i**2
    biased_var_estimate = np.sum(cov_i_hoch2, axis=0)

    bias_correction = n / B * (m - 1) / m * np.var(T_N_b, axis=0)

    return biased_var_estimate, bias_correction


def simulation(seed:int, tau:float, data_generation_weibull_parameters:dict, X_pred_point:pd.DataFrame, params_rf:dict, B_first_level: int ,
               ijk_std_calc: bool, boot_std_calc: bool, jk_ab_calc: bool,  train_models: bool ):

    ########################################### Dataset Creation ############################################################################################
    data_generation_weibull_parameters['seed'] = seed
    df_train, df_test\
        , n_events_after_cut_train, portion_censored_after_cut_train\
        , n_events_after_cut_test, portion_censored_after_cut_test = create_train_test_data(params=data_generation_weibull_parameters)

    #train
    portion_events_after_cut_train = n_events_after_cut_train/df_train.shape[0]
    portion_no_events_after_cut_train = (1-n_events_after_cut_train/df_train.shape[0]-portion_censored_after_cut_train)

    #test
    portion_events_after_cut_test = n_events_after_cut_test/df_test.shape[0]
    portion_no_events_after_cut_test = (1-n_events_after_cut_test/df_test.shape[0]-portion_censored_after_cut_test)

    if train_models == True:
        ############################################ Weibull Modell ############################################################################################
        # Fitten des Weibull Modells
        aft = WeibullAFTFitter()
        aft.fit(df=df_train.drop(['weights_ipcw', 'survived'], axis=1), 
                duration_col='time', 
                event_col='event')
        
        # Evaluation auf Testdaten
        y_pred = aft.predict_survival_function(df=df_test.drop(['weights_ipcw', 'survived','time','event'], axis=1),
                                            times = tau).iloc[0].tolist()
        
        wb_mse_ipcw = ipc_weighted_mse(y_true=df_test['survived'].values, 
                                    y_pred=y_pred, 
                                    sample_weight=df_test['weights_ipcw'])
        

        df_test2 = df_test.copy()
        df_test2 = df_test2[df_test2['time']<=df_train['time'].max()]  
        wb_cindex_ipcw, concordant, discordant, tied_risk, tied_time = concordance_index_ipcw(
                survival_train = Surv.from_arrays(event=df_train['event'], time=df_train['time']),
                survival_test  = Surv.from_arrays(event=df_test2['event'], time=df_test2['time']),
                estimate       =  -aft.predict_expectation(df_test2) )
        
        # Prediction für X_erwartung
        wb_y_pred_X_point = aft.predict_survival_function(df=X_pred_point, 
                                                        times = tau).iloc[0].tolist()


        ######################################### Random Forest Modell #########################################################################################
        # Fitten des Random Forest Modells
        params_rf['random_state'] = seed
        clf = RandomForestClassifier(**params_rf)
        clf.fit(    X=df_train.drop(['time', 'event', 'weights_ipcw', 'survived'], axis=1), 
                    y=df_train['survived'], 
                    sample_weight=df_train['weights_ipcw']  )
        # Evaluation auf Testdaten
        rf_mse_ipcw = ipc_weighted_mse( y_true=df_test['survived'].values, 
                                        y_pred=clf.predict_proba(df_test.drop(['weights_ipcw', 'survived','time','event'], axis=1))[:,1], 
                                        sample_weight=df_test['weights_ipcw']   )
        # Prediction für X_erwartung
        rf_y_pred_X_point = clf.predict_proba(X_pred_point)[:,1]

    else:
        wb_mse_ipcw = 0.
        wb_cindex_ipcw = 0.
        wb_y_pred_X_point = [0.]
        rf_mse_ipcw = 0.
        rf_y_pred_X_point = [0.]


    #######################################################################################################################################################
    ######################################## Variance Estimation ##########################################################################################
    #######################################################################################################################################################


    
    ### IJK Variance Estimation WEIGHTED ##################################################################################################################
    if ijk_std_calc:
        tnb = np.array([tree.predict_proba(X_pred_point.values)[:, 1][0] for tree in clf.estimators_]) 
        biased_var_estimate, bias_correction = inf_JK_bagged_variance(  N_bi=get_Nbi(clf.estimators_samples_), 
                                                                        T_N_b=tnb,
                                                                        weights=df_train['weights_ipcw']  )
        ijk_var_pred_X_point = biased_var_estimate - bias_correction
    else:
        ijk_var_pred_X_point = 0.


    ### Bootstrap Variance Estimation WEIGHTED  ###########################################################################################################
    if boot_std_calc:
        n = df_train.shape[0]
        rng = np.random.default_rng(seed)
        first_level_boot_indices = rng.choice(np.arange(n), size=(B_first_level, n), replace=True)
        preds = np.zeros(B_first_level)

        clf = RandomForestClassifier(**params_rf)
        kmf = KaplanMeierFitter()
        for b in range(B_first_level):
            ### cut data at tau // ipcw weights ###
            df_train_ = df_train.iloc[first_level_boot_indices[b]]
            kmf.fit(df_train_['time'], event_observed=1-df_train_['event'])
            df_train_ = create_new_dataset_with_ipcw_weights(data=df_train_,t=tau, kmf=kmf)

            clf.set_params(random_state=seed+b)
            clf.fit(X=df_train_.drop(['time', 'event', 'weights_ipcw', 'survived'], axis=1), 
                    y=df_train_['survived'], 
                    sample_weight=df_train_['weights_ipcw'])
            preds[b] = clf.predict_proba(X_pred_point)[:, 1]

        bootstrap_std_pred_X_point = np.std(preds)
    else:
        bootstrap_std_pred_X_point = 0.

    return portion_events_after_cut_train, portion_censored_after_cut_train, portion_no_events_after_cut_train,\
           portion_events_after_cut_test, portion_censored_after_cut_test, portion_no_events_after_cut_test,\
           wb_mse_ipcw, wb_cindex_ipcw, wb_y_pred_X_point, rf_mse_ipcw, rf_y_pred_X_point, ijk_var_pred_X_point, bootstrap_std_pred_X_point


'''    ### Jackkknife after Bootstrap Variance Estimation UN-WEIGHTED ######################################################################################
    if jk_ab_calc:

                # Input prediction sample
        x_pred = np.array([-0.42016338,  0.26413616, -1.64544487, -0.70228821 , 0.51820725]).reshape(1, -1)

        # Precompute predictions for all trees
        tree_preds = np.array([estimator.predict_proba(x_pred)[0, 1] for estimator in rf.estimators_])

        # Cache the estimators' samples array for efficient reuse
        estimators_samples = np.array(rf.estimators_samples_, dtype=object)

        # Prepare a boolean mask for each sample's presence in each estimator's bootstrap
        presence_mask = np.zeros((n_samples, n_estimators), dtype=bool)
        for i, samples in enumerate(estimators_samples):
            # Convert the sample list to a NumPy array of integers
            samples = np.array(samples, dtype=int)
            presence_mask[samples, i] = True

        # Calculate the theta_is efficiently
        theta_is = []
        for ii in range(n_samples):
            # Use the presence mask to identify trees where sample ii is not in the bootstrap sample
            indices_without_ii = np.where(~presence_mask[ii])[0]

            # Skip samples that are included in all or none of the trees
            if 0 < len(indices_without_ii) < n_estimators:
                # Calculate the mean prediction for these trees
                theta_is.append(tree_preds[indices_without_ii].mean())

        # Convert theta_is to a NumPy array
        theta_is = np.array(theta_is)

        # Compute the final variance estimates
        theta = rf.predict_proba(x_pred)
        var_jka_biased = np.sum((theta_is - theta[0, 1]) ** 2) * (n_samples - 1) / n_samples

        var_jka_correction = (np.exp(1) - 1) * (n_samples / n_estimators) * np.var(tree_preds)
        var_jka_unbiased = var_jka_biased - var_jka_correction

        var_jka_unbiased

    else:
        var_jka_unbiased = 0.
'''
        

   





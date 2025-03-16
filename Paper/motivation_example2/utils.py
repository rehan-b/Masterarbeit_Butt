import numpy as np
from typing import Tuple, Dict
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import pandas as pd
import os


def step_function(x):
    y_true = np.piecewise(
        x,
        [
            x < 0.35,
            (x >= 0.35) & (x < 0.45),
            (x >= 0.45) & (x < 0.55),
            (x >= 0.55) & (x < 0.65),
            x >= 0.65,
        ],
        [0.0, 0.7, 1.4, 0.7, 0.0],
    )
    return y_true


def generate_response(  x_train: np.ndarray, 
                        loc = 0,
                        var=0.25,
                        seed: int = None):

    rng = np.random.default_rng(seed)
    n_x = x_train.shape[0]

    noise = rng.normal(loc=loc, scale=np.sqrt(var), size=n_x)
    y_true = step_function(x_train)
    y_train = y_true + noise

    return y_train


def create_bootstrap_indices_and_Nbi(
    n: int, B: int, seed: int = None, boot_weights: np.ndarray = None
):
    if boot_weights is None:
        rng = np.random.default_rng(seed)
        boot_indices = rng.choice(np.arange(n), size=(B, n), replace=True)
        boot_counts = np.apply_along_axis(
            lambda x: np.bincount(x, minlength=n), axis=1, arr=boot_indices
        )
        return boot_indices, boot_counts

    else:
        rng = np.random.default_rng(seed)
        boot_indices = rng.choice(np.arange(n), size=(B, n), p=boot_weights, replace=True)
        boot_counts = np.apply_along_axis(
            lambda x: np.bincount(x, minlength=n), axis=1, arr=boot_indices
        )
        return boot_indices, boot_counts


def bagging_decision_trees(
    x_train: np.ndarray,
    y_train: np.ndarray,
    new_data: np.ndarray,
    B: int,
    dt_args: Dict,
    seed: int = None,
    boot_weights: np.ndarray = None,
):
    n = x_train.shape[0]
    n_pred = new_data.shape[0]
    tree_predictions_b = np.zeros(shape=(B, n_pred))
    indices_list, N_bi = create_bootstrap_indices_and_Nbi(
        n=n, B=B, seed=seed, boot_weights=boot_weights
    )

    x_reshaped = x_train.reshape(-1, 1)
    new_data_reshaped = new_data.reshape(-1, 1)

    for b in range(B):
        tree_model = DecisionTreeRegressor(**dt_args)
        tree_model.fit(x_reshaped[indices_list[b]], y_train[indices_list[b]])
        tree_predictions_b[b] = tree_model.predict(new_data_reshaped)

    return tree_predictions_b, N_bi


def ij_variance( N_bi: np.ndarray, T_N_b: np.ndarray):

    B, n = N_bi.shape
    T_N_b_mean = np.mean(T_N_b, axis=0)

    cov_i = ((N_bi - 1).T @ (T_N_b - T_N_b_mean)) / B
    cov_i_hoch2 = cov_i**2
    biased_var_estimate = np.sum(cov_i_hoch2, axis=0)

    bias_correction = (n  / B) * np.var(T_N_b, axis=0, ddof=1)
    return biased_var_estimate, bias_correction



################################################################################### Funktion noch überprüfen
def ij_w_variance( N_bi: np.ndarray, T_N_b: np.ndarray, boot_weights: np.ndarray):

    B, n = N_bi.shape
    T_N_b_mean = np.mean(T_N_b, axis=0)
    n_plus = np.sum(boot_weights > 0)

    cov_i = ((N_bi - n * boot_weights.reshape(1,-1)).T @ (T_N_b - T_N_b_mean)) / B
    cov_i_hoch2 = cov_i**2
    array = cov_i_hoch2/((boot_weights.reshape(-1,1))**2)

    biased_var_estimate = np.sum(array[~np.isnan(array) & ~np.isinf(array)], axis=0) * (1/n_plus**2)


    biased_var_estimate = np.sum(cov_i_hoch2, axis=0)

    bias_correction =  (1/n_plus**2)  * np.var(T_N_b, axis=0, ddof=1)* n / B * np.sum( ( 1 / (boot_weights[boot_weights > 0] ) ) -1) 

    return biased_var_estimate, bias_correction
################################################################################### Funktion noch überprüfen

def simulate_bagging_and_variance(
    n: int,
    B: int,
    new_data: np.ndarray,
    simulation_index: int,
    seed: int,
    dt_args: Dict,
    boot_weights: np.ndarray = None,
    ij_variance_calc: bool = True,
    ij_w_variance_calc: bool = False,
    fixed_x: bool = False,
):
    
    # Seed adjustment
    adjusted_seed = seed + simulation_index
    rng = np.random.default_rng(adjusted_seed)

    # Generate data 
    if fixed_x:
        x_train = np.linspace(0, 1, n) 
    else:
        x_train = rng.uniform(0, 1, n)
    y_train = generate_response(x_train = x_train, seed=adjusted_seed)


    # Perform bagging
    tree_predictions_b, N_bi = bagging_decision_trees(
        x_train=x_train,
        y_train=y_train,
        new_data=new_data,
        B=B,
        dt_args=dt_args,
        seed=adjusted_seed,
        boot_weights=boot_weights,
    )
    bl_predictions = tree_predictions_b.mean(axis=0)

    # Calculate the variance
    if ij_variance_calc:
        biased_var_estimate, bias_correction = ij_variance(N_bi=N_bi, T_N_b=tree_predictions_b)
    elif ij_w_variance_calc:
        biased_var_estimate, bias_correction = ij_w_variance(N_bi=N_bi, T_N_b=tree_predictions_b, boot_weights=boot_weights)
    else:
        biased_var_estimate = np.zeros(n)
        bias_correction = np.zeros(n)

    return bl_predictions, biased_var_estimate, bias_correction




def save_results_png(
    new_data: np.ndarray,
    bagged_preds: np.ndarray,
    est_vars_biased: np.ndarray,
    bias_correction: np.ndarray,
    y_lim: Tuple[float, float] = [0,0.35],
    folder_name: str = "test_folder",
    std_estimator_name: str = "IJK-WAB-U",

    show_only_plot: bool = False,
    show_only_unbiased: bool = True,

):
    true_std = bagged_preds.std(axis=0, ddof=1)

    unbiased_std_estimate = np.where((est_vars_biased - bias_correction)<0, 0, (est_vars_biased - bias_correction))**0.5
    unbiased_std_estimate_mean = unbiased_std_estimate.mean(axis=0)

    biased_std_mean = (np.where(est_vars_biased <0, 0, est_vars_biased)**0.5).mean(axis=0)

    lower_bound = unbiased_std_estimate_mean - unbiased_std_estimate.std(axis=0)
    upper_bound = unbiased_std_estimate_mean + unbiased_std_estimate.std(axis=0)
    plt.rcParams["text.usetex"] = True
    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(new_data, true_std, label="Emp. std")
    plt.plot(
        new_data,
        unbiased_std_estimate.mean(axis=0),
        label=f"{std_estimator_name}",
        alpha=0.6,
    )
    if not show_only_unbiased:
        plt.plot(new_data, biased_std_mean, label=f"Mean estimated std - {std_estimator_name}", alpha=0.4)

    plt.fill_between(
        new_data,
        lower_bound,
        upper_bound,
        color="b",
        alpha=0.2,
        label="±1 std",
    )
    plt.xlabel("x")
    plt.ylabel("std")

    if y_lim is not None:
        plt.ylim(y_lim)
    plt.grid(True)

    plt.legend()

    if show_only_plot:
        plt.show()

    else:
        directory_path = f"./results/{folder_name}"
        os.makedirs(directory_path, exist_ok=True)

        
        plt.savefig(
            f"{directory_path}/plot.png",
            dpi=300, bbox_inches='tight'
        )

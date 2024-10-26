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


def generate_data(
    x: np.ndarray, seed: int = None, noise_variance=0.25
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_x = x.shape[0]

    noise = rng.normal(loc=0, scale=np.sqrt(noise_variance), size=n_x)
    y_true = step_function(x)
    y_noisy = y_true + noise

    return x, y_true, y_noisy


def create_bootstrap_indices_and_Nbi(
    n: int, B: int, seed: int = None, weights: np.ndarray = None
):
    if weights is None:
        rng = np.random.default_rng(seed)
        boot_indices = rng.choice(np.arange(n), size=(B, n), replace=True)
        boot_counts = np.apply_along_axis(
            lambda x: np.bincount(x, minlength=n), axis=1, arr=boot_indices
        )
        return boot_indices, boot_counts
    else:
        rng = np.random.default_rng(seed)
        boot_indices = rng.choice(np.arange(n), size=(B, n), p=weights, replace=True)
        boot_counts = np.apply_along_axis(
            lambda x: np.bincount(x, minlength=n), axis=1, arr=boot_indices
        )
        return boot_indices, boot_counts


def bagging_decision_trees(
    x: np.ndarray,
    y_noisy: np.ndarray,
    new_data: np.ndarray,
    B: int,
    dt_args: Dict,
    seed: int = None,
    weights: np.ndarray = None,
):
    n = x.shape[0]
    n_pred = new_data.shape[0]
    tree_predictions_b = np.zeros(shape=(B, n_pred))
    indices_list, N_bi = create_bootstrap_indices_and_Nbi(
        n=n, B=B, seed=seed, weights=weights
    )

    x_reshaped = x.reshape(-1, 1)
    new_data_reshaped = new_data.reshape(-1, 1)

    for b in range(B):
        tree_model = DecisionTreeRegressor(**dt_args)
        tree_model.fit(x_reshaped[indices_list[b]], y_noisy[indices_list[b]])
        tree_predictions_b[b] = tree_model.predict(new_data_reshaped)

    return tree_predictions_b, N_bi


def inf_JK_bagged_variance(
    N_bi: np.ndarray, T_N_b: np.ndarray, weights: np.ndarray = None, m: int = None
):
    B, n = N_bi.shape
    T_N_b_mean = np.mean(T_N_b, axis=0)

    if weights is None:
        cov_i = ((N_bi - 1).T @ (T_N_b - T_N_b_mean)) / B
        cov_i_hoch2 = cov_i**2
        biased_var_estimate = np.sum(cov_i_hoch2, axis=0)

        bias_correction = ((n - 1) / B) * np.var(T_N_b, axis=0)
        return biased_var_estimate, bias_correction

    else:
        cov_i = ((N_bi - n * weights[0]).T @ (T_N_b - T_N_b_mean)) / B
        cov_i_hoch2 = cov_i**2
        biased_var_estimate = np.sum(cov_i_hoch2, axis=0)

        bias_correction = n / B * (m - 1) / m * np.var(T_N_b, axis=0)

        return biased_var_estimate, bias_correction


def simulate_bagging_and_variance(
    x1: np.ndarray,
    B: int,
    new_data: np.ndarray,
    simulation_index: int,
    seed: int,
    dt_args: Dict,
    weights: np.ndarray = None,
    m: int = None,
    noise_var_for_generating_data=0.25,
    ijk_calculation=True,
):
    adjusted_seed = seed + simulation_index

    x, y_true, y_noisy = generate_data(
        x=x1,
        seed=adjusted_seed,
        noise_variance=noise_var_for_generating_data,
    )

    n = x.shape[0]

    # Perform bagging
    tree_predictions_b, N_bi = bagging_decision_trees(
        x=x,
        y_noisy=y_noisy,
        new_data=new_data,
        B=B,
        dt_args=dt_args,
        seed=adjusted_seed,
        weights=weights,
    )
    bagged_predictions = tree_predictions_b.mean(axis=0)

    if ijk_calculation:
        biased_var_estimate, bias_correction = inf_JK_bagged_variance(
            N_bi=N_bi, T_N_b=tree_predictions_b, weights=weights, m=m
        )
    else:
        biased_var_estimate = np.zeros(n)
        bias_correction = np.zeros(n)

    return bagged_predictions, biased_var_estimate, bias_correction


def save_result_csv(
    seed,
    B,
    args,
    bagged_preds,
    est_vars_biased,
    bias_correction,
    new_data,
    folder_name: str = "test_folder",
    fix_x_points: bool = True,
):
    directory_path = "./results/" + folder_name
    os.makedirs(directory_path, exist_ok=True)

    if fix_x_points:
        name = f"{directory_path}/seed{seed}_nB{B}_fixed_x_{args.items()}"
    else:
        name = f"{directory_path}/seed{seed}_nB{B}_new_x_{args.items()}"

    header = ["pred-x-points"] + new_data.tolist()

    ############################
    combined_data = np.hstack(
        (
            np.array([["sim" + str(x)] for x in range(1, bagged_preds.shape[0] + 1)]),
            bagged_preds,
        )
    )
    df = pd.DataFrame(combined_data)
    df.columns = header
    df.to_csv(name + "bagged_preds.csv", index=False, sep=";")
    ############################

    combined_data = np.hstack(
        (
            np.array([["sim" + str(x)] for x in range(1, bagged_preds.shape[0] + 1)]),
            est_vars_biased,
        )
    )
    df = pd.DataFrame(combined_data)
    df.columns = header
    df.to_csv(name + "ijk_std_biased.csv", index=False, sep=";")
    ############################

    combined_data = np.hstack(
        (
            np.array([["sim" + str(x)] for x in range(1, bagged_preds.shape[0] + 1)]),
            bias_correction,
        )
    )
    df = pd.DataFrame(combined_data)
    df.columns = header
    df.to_csv(name + "bias_correction.csv", index=False, sep=";")
    ############################

    combined_data = np.hstack(
        (
            np.array([["sim" + str(x)] for x in range(1, bagged_preds.shape[0] + 1)]),
            est_vars_biased - bias_correction,
        )
    )
    df = pd.DataFrame(combined_data)
    df.columns = header
    df.to_csv(name + "ijk_std_unbiased.csv", index=False, sep=";")
    ############################


def save_results_png(
    new_data: np.ndarray,
    bagged_preds: np.ndarray,
    est_vars_biased: np.ndarray,
    bias_correction: np.ndarray,
    y_lim: Tuple[float, float] = [0, 0.3],
    folder_name: str = "test_folder",
    n_data_points: np.ndarray = None,
    B: int = None,
    seed: int = None,
    dt_args: Dict = None,
    fixed_x_points: bool = True,
    show_only_plot: bool = False,
    show_only_unbiased: bool = True,
    m: bool = False,
    reduced: bool = False,
):
    n_simulations = bagged_preds.shape[0]

    true_std = bagged_preds.std(axis=0)

    unbiased_std_estimate = (est_vars_biased - bias_correction) ** 0.5
    unbiased_std_estimate_mean = unbiased_std_estimate.mean(axis=0)

    biased_std_mean = (est_vars_biased**0.5).mean(axis=0)

    lower_bound = unbiased_std_estimate_mean - unbiased_std_estimate.std(axis=0)
    upper_bound = unbiased_std_estimate_mean + unbiased_std_estimate.std(axis=0)

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(new_data, true_std, label="Emp. std")
    plt.plot(
        new_data,
        unbiased_std_estimate_mean,
        label="Mean est. std IJK-WAB-U",
        alpha=0.6,
    )
    if not show_only_unbiased:
        plt.plot(new_data, biased_std_mean, label="Mean Est. std IJK-biased", alpha=0.4)
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
        directory_path = "./figures/" + folder_name
        os.makedirs(directory_path, exist_ok=True)

        if fixed_x_points:
            plt.savefig(
                f"{directory_path}/seed{seed}_nB{B}_fixed_x_{dt_args.items()}.png",
                dpi=600,
            )
        else:
            plt.savefig(
                f"{directory_path}/seed{seed}_nB{B}_new_x_{dt_args.items()}.png",
                dpi=600,
            )

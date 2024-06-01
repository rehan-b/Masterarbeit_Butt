import numpy as np
from typing import Tuple, Dict
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import time
import pandas as pd


def save_results_png(
    new_data: np.ndarray,
    bagged_preds: np.ndarray,
    est_vars: np.ndarray,
    n_data_points: np.ndarray = None,
    B: int = None,
    seed: int = None,
    dt_args: Dict = None,
    fixed_x_points: bool = None,
    show_only_plot: bool = False,
):
    """
    Save the results as a PNG file.
    """
    n_simulations = bagged_preds.shape[0]

    # Calculate true variance of bagged predictions
    true_variances = bagged_preds.var(axis=0, ddof=1)
    est_variances_mean = est_vars.mean(axis=0)
    est_variances_std = est_vars.std(axis=0, ddof=1)
    
    lower_bound = np.maximum(est_variances_mean - est_variances_std, 0)
    upper_bound = est_variances_mean + est_variances_std

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(new_data, true_variances, label="True Variance")
    plt.plot(new_data, est_variances_mean, label="Mean Est. Variance", alpha=0.6)
    plt.fill_between(
        new_data,
        lower_bound,
        upper_bound,
        color="b",
        alpha=0.2,
        label="±1 std",
    )
    plt.title("True Variance of Bagged Predictions Across Simulated Datasets")
    plt.xlabel("x")
    plt.ylabel("Variance")
    #plt.ylim(0.0, 0.06)
    plt.grid(True)

    plt.legend()

    if show_only_plot:
        plt.show()
    else:
        plt.text(
            0.05,
            0.04,
            f"data_points = {n_data_points}\nsimulations = {n_simulations}\nbootstrap(B) = {B}\nfixed_x={fixed_x_points}",
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.5),
        )
        if fixed_x_points:
            plt.savefig(
                f"figure2_wager/figures/figure2_seed{seed}_nB{B}_fixed_x_{dt_args.items()}.png",
                dpi=600,
            )
        else:
            plt.savefig(
                f"figure2_wager/figures/figure2_seed{seed}_nB{B}_{dt_args.items()}.png",
                dpi=600,
            )
        # {int(time.time())}


def inf_JK_bagged_variance(N_bi, T_N_b, weights):
    """
    Calculate the Jackknife-Infenitesimal Variance for bagged learners.

    Parameters:
    - N_bi (numpy.ndarray): The input data matrix of shape (B, n).
    - T_N_b (numpy.ndarray): The predictions matrix of shape (B, n_preds).
    - weights (numpy.ndarray): The weights of the data points for bootstrapping. 

    Returns:
    - var_inf_JK_U (numpy.ndarray): The estimated Jackknife-Infenitesimal Variance for bagged learners.
    """

    B, n = N_bi.shape
    n_preds = T_N_b.shape[1]
    T_N_star_mean = np.mean(T_N_b, axis=0)

    # Initialize the covariance matrix
    cov_i = np.zeros((n, n_preds))

    # Calculate deviations once to avoid recalculations
    N_bi_dev = N_bi - 1
    T_N_b_dev = T_N_b - T_N_star_mean

    # Fill the covariance matrix using vectorized operations
    cov_i = (1 / weights[:, None]) * (N_bi_dev.T @ T_N_b_dev) / (B - 1)
    cov_vector = np.sum(cov_i**2, axis=0) / n**2

    # Bias correction
    bias_correction = (B / (n * (B - 1) ** 2)) * np.var(T_N_b, axis=0, ddof=1) * np.sum((1 - weights) / weights)

    # Estimate of Jackknife-Infenitesimal Variance for bagged learners
    var_inf_JK_U = cov_vector - bias_correction
    return var_inf_JK_U


def inf_JK_bagged_variance_simple(N_bi: np.ndarray, T_N_b: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Calculate the Jackknife-Infenitesimal Variance for bagged learners.

    Parameters:
    - N_bi (numpy.ndarray): The input data matrix of shape (B, n).
    - T_N_b (numpy.ndarray): The predictions matrix of shape (B, n_preds).
    - weights (numpy.ndarray): The weights of the data points for bootstrapping. 

    Returns:
    - var_inf_JK_U (numpy.ndarray): The estimated Jackknife-Infenitesimal Variance for bagged learners.

    """

    B, n = N_bi.shape
    n_preds = T_N_b.shape[1]
    T_N_star_mean = np.mean(T_N_b, axis=0)

    # Initialize the covariance matrix
    cov_i = np.zeros((n, n_preds))

    # Fill the covariance matrix
    for i in range(n):
        for pred in range(n_preds):

            cov_i[i, pred] = (1/weights[i]) * np.dot(
                (N_bi[:, i] - 1), (T_N_b[:, pred] - T_N_star_mean[pred])
            ) / (B - 1)
    cov_vector = (1/n**2) * np.sum(cov_i**2, axis=0)

    # Bias correction
    bias_correction = ( (B) / (n * (B - 1) ** 2) ) * np.var(
        T_N_b, axis=0, ddof=1
    ) * np.sum((1 - weights) / weights)
        
    # Estimate of Jackknife-Infenitesimal Variance for bagged learners
    var_inf_JK_U = cov_vector - bias_correction
    return var_inf_JK_U


def create_bootstrap_indices_and_Nbi(
    n: int, B: int, seed: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates bootstrap indices and counts.

    Parameters:
    - n (int): The number of observations.
    - B (int): The number of bootstrap samples to generate.
    - seed (int, optional): The seed value for random number generation. Default is None.

    Returns:
    - boot_indices (np.ndarray): An array of shape (B, n_data_points) containing the bootstrap indices.
    - boot_counts  (np.ndarray): An array of shape (B, n_data_points) containing the counts.

    Example:
    >>> indices_list, counts = create_bootstrap_indices_and_Nbi(5, 3, seed=42)
    >>> indices_list
    array([[3, 4, 2, 4, 4],
           [1, 2, 2, 2, 4],
           [3, 2, 4, 1, 3]])
    >>> counts
    array([[0, 0, 1, 1, 3],
           [0, 1, 3, 0, 1],
           [0, 1, 1, 2, 1]])
    """
    rng = np.random.default_rng(seed)
    boot_indices = rng.integers(0, n, size=(B, n))
    boot_counts = np.apply_along_axis(
        lambda x: np.bincount(x, minlength=n), axis=1, arr=boot_indices
    )
    return boot_indices, boot_counts


def generate_data(
    n: int, seed: int = None, noise_variance=0.25, fix_points=False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic data for a regression problem.

    Args:
        n (int): The number of data points to generate.
        seed (int, optional): The seed value for the random number generator. Defaults to None.
        noise_variance (float, optional): The variance of the Gaussian noise added to the true labels. Defaults to 0.25.
        fix_points (bool, optional): If True, fix the x-coordinates of the data points. Defaults to False.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing three arrays: \n
            - x: The x-coordinates of the data points. \n
            - y true: The true labels without noise. \n
            - y noisy: The true labels with added Gaussian noise.
    """

    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, n) if fix_points else rng.uniform(0, 1, n)

    # Step function
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

    noise = rng.normal(loc=0, scale=np.sqrt(noise_variance), size=n)
    y_noisy = y_true + noise

    return x, y_true, y_noisy


def bagging_decision_trees(
    x: np.ndarray,
    y_noisy: np.ndarray,
    new_data: np.ndarray,
    B: int,
    dt_args: Dict,
    seed: int = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Train and predict using bagging with decision trees.

    Args:
        x (np.ndarray): Training data features.
        y_noisy (np.ndarray): Training data targets with noise.
        new_data (np.ndarray): New data points to predict.
        B (int): Number of bootstrap samples and trees.
        dt_args (dict): Arguments for the DecisionTreeRegressor.
        seed (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing: \n
            - tree predictions b (np.ndarray): Predictions from each tree of shape=(B, n pred). \n
            - N bi (np.ndarray): Bootstrap sample counts of shape=(B, n).
    """
    n = x.shape[0]
    n_pred = new_data.shape[0]
    tree_predictions_b = np.zeros(shape=(B, n_pred))
    indices_list, N_bi = create_bootstrap_indices_and_Nbi(n=n, B=B, seed=seed)

    x_reshaped = x.reshape(-1, 1)
    new_data_reshaped = new_data.reshape(-1, 1)

    for b in range(B):
        tree_model = DecisionTreeRegressor(**dt_args)
        tree_model.fit(x_reshaped[indices_list[b]], y_noisy[indices_list[b]])
        tree_predictions_b[b] = tree_model.predict(new_data_reshaped)

    return tree_predictions_b, N_bi


def simulate_bagging_and_variance(
    n: int,
    B: int,
    new_data: np.ndarray,
    simulation_index: int,
    seed: int,
    dt_args: Dict,
    weights: np.ndarray ,
    noise_var_for_generating_data=0.25,
    fix_x_points=False,
    ijk_calculation=False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulates bagging and calculates variance for a given set of parameters.

    Args:
        n (int): Number of data points.
        B (int): Number of bootstrap samples.
        new_data (np.ndarray): New data points for prediction.
        simulation_index (int): Index of the simulation.
        seed (int): Seed for random number generation.
        dt_args (Dict): Dictionary of arguments for decision tree.
        weights (np.ndarray): Weights for bootstrapping.
        noise_var_for_generating_data (float, optional): Variance of noise for generating data. Defaults to 0.25.
        fix_x_points (bool, optional): Whether to fix x points. Defaults to False.
        ijk_calculation (bool, optional): Whether to calculate infinitesimal jackknife variance. Defaults to False.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple containing bagged predictions and estimated variances.
    """

    adjusted_seed = seed + simulation_index

    x, y_true, y_noisy = generate_data(
        n=n,
        seed=adjusted_seed,
        noise_variance=noise_var_for_generating_data,
        fix_points=fix_x_points,
    )

    if fix_x_points:
        x = np.linspace(0, 1, n)

    # Perform bagging
    tree_predictions_b, N_bi = bagging_decision_trees(
        x=x,
        y_noisy=y_noisy,
        new_data=new_data,
        B=B,
        dt_args=dt_args,
        seed=adjusted_seed,
    )
    bagged_predictions = tree_predictions_b.mean(axis=0)

    if ijk_calculation:
        est_variances = inf_JK_bagged_variance(
            N_bi=N_bi, T_N_b=tree_predictions_b, weights= weights
        )
    else:
        est_variances = np.zeros(n)

    return bagged_predictions, est_variances


def save_result_csv(fix_x_points, seed, B, args, bagged_preds, est_vars, new_data):

    if fix_x_points:
        name = f"figure2_wager/results/seed{seed}_nB{B}_fixed_x_{args.items()}"
    else:
        name = f"figure2_wager/results/seed{seed}_nB{B}_new_x_{args.items()}"

    header = ["pred-x-points"] + new_data.tolist()
    combined_data = np.hstack(
        (
            np.array([["sim" + str(x)] for x in range(1, bagged_preds.shape[0] + 1)]),
            bagged_preds,
        )
    )
    df = pd.DataFrame(combined_data)
    df.columns = header
    df.to_csv(name + "bagged_preds.csv", index=False, sep=";")

    combined_data = np.hstack(
        (
            np.array([["sim" + str(x)] for x in range(1, bagged_preds.shape[0] + 1)]),
            est_vars,
        )
    )
    df = pd.DataFrame(combined_data)
    df.columns = header
    df.to_csv(name + "est_vars.csv", index=False, sep=";")


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

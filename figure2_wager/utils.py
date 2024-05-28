import numpy as np
from typing import Tuple
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import time


def create_bootstrap_indices_and_Nbi(
    n_data_points: int, B: int, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates bootstrap indices and counts.

    Parameters:
    - n_data_points (int): The number of data points.
    - B (int): The number of bootstrap samples to generate.
    - seed (int, optional): The seed value for random number generation. Default is None.

    Returns:
    - indices_list (np.ndarray): An array of shape (B, n_data_points) containing the bootstrap indices.
    - counts (np.ndarray): An array of shape (B, n_data_points) containing the counts.

    Example:
    >>> indices_list, counts = create_bootstrap_indices_and_Nbi(5, 3, seed=42)
    >>> indices_list
    (array([[3, 4, 2, 4, 4],
        [1, 2, 2, 2, 4],
        [3, 2, 4, 1, 3]])
    >>> counts
    array([[0, 0, 1, 1, 3],
        [0, 1, 3, 0, 1],
        [0, 1, 1, 2, 1]]))
    """
    np.random.seed(seed=seed)
    indices_list = np.random.choice(
        n_data_points, size=(B, n_data_points), replace=True
    )
    counts = np.zeros(shape=(B, n_data_points), dtype=int)
    for x_i in range(n_data_points):
        counts[:, x_i] = np.sum(indices_list == x_i, axis=1)
    return indices_list, counts


def bagging_decision_trees(
    x: np.ndarray, y_noisy: np.ndarray,new_data: np.ndarray, B: int, 
    max_leaf_nodes: int, seed: int, min_samples_leaf: int= 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns: \n
    tree_predictions_b(B, n_data_points) , \n
    N_bi(B, n_data_points)"""

    n_data_points = x.shape[0]
    n_pred_points = new_data.shape[0]
    tree_predictions_b = np.zeros(shape=(B, n_pred_points))
    indices_list, N_bi = create_bootstrap_indices_and_Nbi(
        n_data_points=n_data_points, B=B, seed=seed
    )

    for b in range(B):
        tree_model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes,min_samples_leaf=min_samples_leaf)
        tree_model.fit(
            X=x[indices_list[b]].reshape(-1, 1), y=y_noisy[indices_list[b]]
        )
        tree_predictions_b[b, :] = tree_model.predict(X=new_data.reshape(-1, 1))

    return tree_predictions_b, N_bi


def step_function(x):
    """
    Returns the step function value for the given input.

    Parameters:
    x (float or array-like): The input value(s) for which the step function value is calculated.

    Returns:
    float or array-like: The step function value(s) corresponding to the input value(s).
    """
    return np.where(
        x < 0.35,
        0.0,
        np.where(x < 0.45, 0.7, np.where(x < 0.55, 1.4, np.where(x < 0.65, 0.7, 0.0))),
    )


def generate_data(
    n_data_points: int, seed: int, noise_variance=0.25
):
    """Generates noisy data based on the true function."""
    np.random.seed(seed=seed)
    x = np.random.uniform(0, 1, n_data_points)
    #x = np.linspace(0, 1, n_data_points)
    y_true = step_function(x)
    noise = np.random.normal(loc=0, scale=np.sqrt(noise_variance), size=n_data_points)
    return x , y_true, y_true + noise


def inf_JK_bagged_variance(
    N_bi: np.ndarray, T_N_b: np.ndarray, chunk_size: int
) -> np.ndarray:
    """
    Calculate the Jackknife-Infenitesimal Variance for bagged learners.

    Args:
        N_bi (ndarray): The input data matrix of shape (B, n_data_points).
        T_N_b (ndarray): The predictions matrix of shape (B, n_preds).
        chunk_size (int): The size of each chunk for calculating the covariance matrix.

    Returns:
        ndarray: The Jackknife-Infenitesimal Variance for bagged learners of shape (n_preds,).

    """
    B, n_data_points = N_bi.shape
    n_preds = T_N_b.shape[1]
    T_N_star_mean = np.mean(T_N_b, axis=0)

    # Initialize the covariance matrix
    cov_i = np.zeros(shape=(n_data_points, n_preds))

    # Fill the covariance matrix
    # Calculate in chunks to avoid memory issues
    for start in range(0, B, chunk_size):
        end = min(start + chunk_size, B)
        chunk_N_bi = N_bi[start:end]
        chunk_T_N_b = T_N_b[start:end]
        cov_i += np.sum(
            (chunk_N_bi[:, :, None] - 1) * (chunk_T_N_b[:, None, :] - T_N_star_mean),
            axis=0,
        )
    cov_i /= B - 1
    cov_vector = np.sum(cov_i**2, axis=0)

    # Bias correction
    bias_correction = (((n_data_points - 1) * B) / (B - 1) ** 2) * np.var(
        T_N_b, axis=0, ddof=1
    )

    # Estimate of Jackknife-Infenitesimal Variance for bagged learners
    var_inf_JK_U = cov_vector - bias_correction
    return var_inf_JK_U


def inf_JK_bagged_variance_simple(N_bi: np.ndarray, T_N_b: np.ndarray) -> np.ndarray:
    """
    Calculate the Jackknife-Infenitesimal Variance for bagged learners.

    Parameters:
    - N_bi (numpy.ndarray): The input data matrix of shape (B, n_data_points).
    - T_N_b (numpy.ndarray): The predictions matrix of shape (B, n_preds).

    Returns:
    - var_inf_JK_U (numpy.ndarray): The estimated Jackknife-Infenitesimal Variance for bagged learners.

    """

    B, n_data_points = N_bi.shape
    n_preds = T_N_b.shape[1]
    T_N_star_mean = np.mean(T_N_b, axis=0)

    # Initialize the covariance matrix
    cov_i = np.zeros((n_data_points, n_preds))

    # Fill the covariance matrix
    for i in range(n_data_points):
        for pred in range(n_preds):

            cov_i[i, pred] = np.dot(
                (N_bi[:, i] - 1), (T_N_b[:, pred] - T_N_star_mean[pred])
            ) / (B - 1)
    cov_vector = np.sum(cov_i**2, axis=0)

    # Bias correction
    bias_correction = (((n_data_points - 1) * B) / (B - 1) ** 2) * np.var(
        T_N_b, axis=0, ddof=1
    )
    # Estimate of Jackknife-Infenitesimal Variance for bagged learners
    var_inf_JK_U = cov_vector - bias_correction
    return var_inf_JK_U


def simulate_bagging_and_variance(
    new_data: np.ndarray,
    n_data_points: int,
    B: int,
    simulation_index: int,
    seed: int,
    noise_variance_for_y=0.25,
    max_leaf_nodes=5,
    chunk_size=250,
    ijk_calculation=False,
    min_samples_leaf: int= 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate bagging and calculate variance for a single run.

    Parameters:
        x_points (array-like): The x-coordinates of the data points.
        y_true (array-like): The true y-values corresponding to the x-coordinates.
        B (int): The number of bootstrap samples to generate.
        simulation_index (int): The index of the simulation.
        seed (int): The seed value for random number generation.
        noise_variance_for_y (float, optional): The variance of the noise added to y-values. Defaults to 0.25.
        max_leaf_nodes (int, optional): The maximum number of leaf nodes in the decision trees. Defaults to 5.
        chunk_size (int, optional): The size of chunks used for infinitesimal jackknife variance calculation. Defaults to 250.

    Returns:
        tuple: A tuple containing the bagged predictions and estimated variances.

    """

    x, y_true, y_noisy = generate_data(
        n_data_points=n_data_points,
        seed=seed + simulation_index,
        noise_variance=noise_variance_for_y,
    )

    # Perform bagging
    tree_predictions_b, N_bi = bagging_decision_trees(
        x=x,
        y_noisy=y_noisy,
        new_data=new_data,
        B=B,
        max_leaf_nodes=max_leaf_nodes,
        seed=seed + simulation_index,
        min_samples_leaf=min_samples_leaf,
    )
    bagged_predictions = tree_predictions_b.mean(axis=0)

    if ijk_calculation:
        # Calculate infinitesimal jackknife variance
        est_variances = inf_JK_bagged_variance(
            N_bi=N_bi, T_N_b=tree_predictions_b, chunk_size=chunk_size
        )
    else: est_variances = np.zeros(n_data_points)

    return bagged_predictions, est_variances


def save_results_png(
    new_data: np.ndarray,
    true_variances: np.ndarray,
    est_variances_mean: np.ndarray,
    est_variances_std: np.ndarray,
    n_data_points: np.ndarray,
    n_simulations: int,
    B: int,
    seed: int,
    noise_variance 
):
    """
    Save the results plot as a PNG file.

    Args:
        new_data (array-like): The x-axis values to predict.
        true_variances (array-like): The true variances.
        est_variances_mean (array-like): The mean estimated variances.
        est_variances_std (array-like): The standard deviation of estimated variances.
        n_data_points (int): The number of data points.
        n_simulations (int): The number of simulations.
        B (int): The bootstrap value.
        seed (int): The seed value.

    Returns:
        None
    """

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(new_data, true_variances, label="True Variance")
    plt.plot(new_data, est_variances_mean, label="Mean Est. Variance")
    plt.fill_between(
        new_data,
        est_variances_mean - est_variances_std,
        est_variances_mean + est_variances_std,
        color="b",
        alpha=0.2,
        label="±1 std",
    )
    plt.title("True Variance of Bagged Predictions Across Simulated Datasets")
    plt.xlabel("x")
    plt.ylabel("Variance")
    #plt.ylim(-0.02, 0.25)
    plt.grid(True)  

    plt.text(
        0.05,
        0.05,
        f"data_points = {n_data_points}\nsimulations = {n_simulations}\nbootstrap(B) = {B}",
        fontsize=12,
        bbox=dict(facecolor="white", alpha=0.5),
    )
    plt.legend()
    plt.savefig(
        f"figure2_wager/figures/figure2_nB{B}_nsim{n_simulations}_noise{noise_variance}.png"
    )
    #{int(time.time())}

import time
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm  

# Constants
NOISE_VARIANCE = 0.25
MAX_LEAF_NODES = 5
CHUNK_SIZE = 250

def step_function(x):
    """Defines the true step function."""
    return np.where(x < 0.35, 0, np.where(x < 0.45, 0.7, np.where(x < 0.55, 1.4, np.where(x < 0.65, 0.7, 0))))

def generate_data(x_points, y_true, noise_variance=NOISE_VARIANCE, seed=None):
    """Generates noisy data based on the true function."""
    np.random.seed(seed)
    noise = np.random.normal(0, np.sqrt(noise_variance), len(x_points)).astype(np.float32)
    return y_true + noise

def create_bootstrap_indices_and_Nbi(n_data_points, n_bootstrap, seed=None):
    """Creates bootstrap indices and counts."""
    np.random.seed(seed)
    indices_list = np.random.choice(n_data_points, (n_bootstrap, n_data_points), replace=True)
    counts = np.zeros((n_bootstrap, n_data_points), dtype=int)
    for x_i in range(n_data_points):
        counts[:, x_i] = np.sum(indices_list == x_i, axis=1)
    return indices_list, counts

def bagging_decision_trees(x_points, y_noisy, n_bootstrap, max_leaf_nodes=MAX_LEAF_NODES, seed=None):
    """Performs bagging with decision trees."""
    n_data_points = x_points.shape[0]
    tree_predictions_b = np.zeros((n_bootstrap, n_data_points), dtype=np.float32)
    indices_list, N_bi = create_bootstrap_indices_and_Nbi(n_data_points=n_data_points, n_bootstrap=n_bootstrap, seed=seed)

    for b in range(n_bootstrap):
        tree_model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes)
        tree_model.fit(x_points[indices_list[b]].reshape(-1, 1), y_noisy[indices_list[b]])
        tree_predictions_b[b, :] = tree_model.predict(x_points.reshape(-1, 1))

    return tree_predictions_b, N_bi

def inf_JK_bagged_variance(N_bi, tree_predictions_b, chunk_size=CHUNK_SIZE):
    """Calculates the infinitesimal jackknife variance estimate."""
    n_bootstrap, n_data_points = N_bi.shape
    n_preds = tree_predictions_b.shape[1]

    N_star_mean = np.mean(N_bi, axis=0).astype(np.float32)
    T_N_star_mean = np.mean(tree_predictions_b, axis=0).astype(np.float32)

    # Initialize the covariance matrix
    cov_matrix = np.zeros((n_data_points, n_preds), dtype=np.float32)
    
    # Calculate in chunks to avoid memory issues
    for start in range(0, n_bootstrap, chunk_size):
        end = min(start + chunk_size, n_bootstrap)
        chunk_N_bi = N_bi[start:end]
        chunk_tree_predictions_b = tree_predictions_b[start:end]
        cov_matrix += np.sum((chunk_N_bi[:, :, None] - N_star_mean) * 
                             (chunk_tree_predictions_b[:, None, :] - T_N_star_mean), axis=0).astype(np.float32)

    cov_matrix /= n_bootstrap

    bias_correction = (n_data_points / n_bootstrap) * np.mean((tree_predictions_b - T_N_star_mean) ** 2, axis=0).astype(np.float32)

    # Calculate infinitesimal jackknife estimate
    bagged_inf_jackknife_est = np.sum(cov_matrix ** 2, axis=0) - bias_correction

    return bagged_inf_jackknife_est

def simulate_bagging_and_variance(x_points, y_true, n_bootstrap, simulation_index, seed):
    """Simulate bagging and calculate variance for a single run."""
    y_noisy = generate_data(x_points, y_true, noise_variance=NOISE_VARIANCE, seed=seed + simulation_index)

    # Perform bagging
    tree_predictions_b, N_bi = bagging_decision_trees(x_points, y_noisy, n_bootstrap, seed=seed + simulation_index)
    bagged_predictions = tree_predictions_b.mean(axis=0)

    # Calculate infinitesimal jackknife variance
    est_variances = inf_JK_bagged_variance(N_bi, tree_predictions_b)

    return bagged_predictions, est_variances

def main():
    """Main function to run the simulation and plotting."""
    # Simulation parameters
    n_data_points = 50
    n_simulations = 1_00
    n_bootstrap = 50  # Adjust as needed
    seed = 62

    # Generate data
    x_points = np.linspace(0, 1, n_data_points, dtype=np.float32)
    y_true = step_function(x_points).astype(np.float32)

    # Arrays to store the predictions and estimated variances
    bagged_predictions_all = np.zeros((n_simulations, n_data_points), dtype=np.float32)
    est_variances_all = np.zeros((n_simulations, n_data_points), dtype=np.float32)

    # Parallelize simulations with progress bar
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(simulate_bagging_and_variance, x_points, y_true, n_bootstrap, i, seed) for i in range(n_simulations)]
        
        for i, future in enumerate(tqdm(futures, desc="Simulations", unit="simulation")):
            bagged_predictions, est_variances = future.result()
            bagged_predictions_all[i, :] = bagged_predictions
            est_variances_all[i, :] = est_variances

    # Calculate true variance of bagged predictions
    true_variances = bagged_predictions_all.var(axis=0, ddof=1)
    est_variances_mean = est_variances_all.mean(axis=0)
    est_variances_std = est_variances_all.std(axis=0, ddof=1)

    print(f"Mean true variance: {round(np.mean(true_variances), 10)}")
    print(f"Mean estimated variance: {round(np.mean(est_variances_mean), 10)}")
    print(f"Min estimated variance: {round(np.min(est_variances_all), 10)}")

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(x_points, true_variances, label='True Variance of Bagged Predictions')
    plt.plot(x_points, est_variances_mean, label='Mean Est. Variance of Bagged Predictions')
    plt.fill_between(x_points, est_variances_mean - est_variances_std, est_variances_mean + est_variances_std, color='b', alpha=0.2, label='±1 std')
    plt.title('True Variance of Bagged Predictions Across Simulated Datasets')
    plt.xlabel('x')
    plt.ylabel('Variance')
    plt.ylim(-0.02, 0.06)
   
    plt.text(0.05, 0.05, f"data_points = {n_data_points}\nsimulations = {n_simulations}\nbootstrap(B) = {n_bootstrap}", fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    plt.legend()
    timex = time.ctime()[4:19]
    timex = timex.replace(":", "_")
    timex = timex.replace(" ", "_")
    # Save figure with name figure2_wager+ n_datapoints+ n_simulations + n_bootstrap + seed 
    plt.savefig(f"figure2_wager_seed{seed}_{timex}.png")

if __name__ == '__main__':
    start_time = time.time()
    main()
    print(f"--- {round((time.time() - start_time), 2)} seconds ---")

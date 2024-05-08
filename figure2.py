import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
import concurrent.futures
import os

# Define the step function as the true function
def step_function(x):
    return np.where(x < 0.35, 0, np.where(x < 0.45, 0.7, np.where(x < 0.55, 1.4, np.where(x < 0.65, 0.7, 0))))

def simulate_and_predict(i, n_data_points, x_values, y_true, n_bootstrap,seed):
    np.random.seed(seed)
    
    noise = np.random.normal(0, 0.5, n_data_points)
    y_noisy = y_true + noise

    # Bagging and prediction
    tree_model = DecisionTreeRegressor(max_leaf_nodes=5)
    tree_predictions = np.zeros((n_bootstrap, n_data_points))
    
    for b in range(n_bootstrap):
        indices = np.random.choice(n_data_points, n_data_points, replace=True)
        tree_model.fit(x_values[indices].reshape(-1, 1), y_noisy[indices])
        tree_predictions[b, :] = tree_model.predict(x_values.reshape(-1, 1))
    
    # Average predictions across bootstrap samples
    return tree_predictions.mean(axis=0)

if __name__ == '__main__':
    start_time = time.time()

    # Simulation parameters
    seed = 42
    np.random.seed(seed)
    n_data_points = 500
    n_simulations = 1_000  # Number of different datasets to simulate
    n_bootstrap = 10_000  # Number of bootstrap samples for bagging
    x_values = np.linspace(0, 1, n_data_points)
    y_true = step_function(x_values)

    # Array to store the predictions from each simulation
    bagged_predictions = np.zeros((n_simulations, n_data_points))

    # Use a ProcessPoolExecutor to parallelize the simulation and prediction
    with concurrent.futures.ProcessPoolExecutor() as executor:
        seeds = [seed + i for i in range(n_simulations)]
        futures = [executor.submit(simulate_and_predict, i, n_data_points, x_values, y_true, n_bootstrap, seeds[i]) for i in range(n_simulations)]
        bagged_predictions = [future.result() for future in concurrent.futures.as_completed(futures)]

    # Calculate true variance of bagged predictions
    true_variances = np.var(bagged_predictions, axis=0)

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, true_variances, label='True Variance of Bagged Predictions')
    plt.title('True Variance of Bagged Predictions Across Simulated Datasets')
    plt.xlabel('x')
    plt.ylabel('Variance')
    plt.legend()
    plt.savefig("figure2.png")

    print(round(np.mean(true_variances),4))
    print("--- %s seconds ---" % (round((time.time() - start_time), 2)))

import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from utils import (
    create_bootstrap_indices_and_Nbi,
    bagging_decision_trees,
    generate_data,
    inf_JK_bagged_variance,
    step_function,
    simulate_bagging_and_variance,
    save_results_png,
)


def main():
    """Main function to run the simulation and saving the results as png."""

    # Constants
    MAX_LEAF_NODES = 5
    CHUNK_SIZE = 250
    sampling_noise = 0.25

    # Simulation parameters
    n_data_points = 500
    n_simulations = 1_000
    B = 500
    seed = 62

    # Generate fixed data points and true function values
    x_points = np.linspace(0, 1, n_data_points)
    y_true = step_function(x=x_points)

    # Arrays to store the predictions and estimated variances
    bagged_predictions_all = np.zeros((n_simulations, n_data_points))
    est_variances_all = np.zeros((n_simulations, n_data_points))

    # Parallelize simulations with progress bar
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                simulate_bagging_and_variance,
                x_points=x_points,
                y_true=y_true,
                B=B,
                simulation_index=i,
                seed=seed,
                noise_variance_for_y=sampling_noise,
                max_leaf_nodes=MAX_LEAF_NODES,
                chunk_size=CHUNK_SIZE,
            )
            for i in range(n_simulations)
        ]

        for i, future in enumerate(
            tqdm(futures, desc="Simulations", unit="simulation")
        ):
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

    save_results_png(
        x_points,
        true_variances,
        est_variances_mean,
        est_variances_std,
        n_data_points,
        n_simulations,
        B,
        seed,
    )


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- runtime: %s minutes ---" % round((time.time() - start_time) / 60, 2))

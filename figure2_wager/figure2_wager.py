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
    sampling_noise_variance = [0.25,0.5,round(np.sqrt(2),4), 2.0]
    min_samples_leaf = 1       # default = 1
    
    for noise in sampling_noise_variance:
        sampling_noise = noise
        
        # Simulation parameters
        n_data_points = 500
        n_simulations = 1_000
        B = 1000
        seed = 62

        # Generate fixed data points for predictions
        new_data = np.linspace(0, 1, n_data_points)

        # Arrays to store the predictions and estimated variances
        bagged_predictions_all = np.zeros((n_simulations, new_data.shape[0]))
        est_variances_all = np.zeros((n_simulations, new_data.shape[0]))

        # Parallelize simulations with progress bar
        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(
                    simulate_bagging_and_variance,
                    new_data=new_data,
                    n_data_points=n_data_points,
                    B=B,
                    simulation_index=i,
                    seed=seed,
                    noise_variance_for_y=sampling_noise,
                    max_leaf_nodes=MAX_LEAF_NODES,
                    chunk_size=CHUNK_SIZE,
                    min_samples_leaf=min_samples_leaf
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
            new_data,
            true_variances,
            est_variances_mean,
            est_variances_std,
            n_data_points,
            n_simulations,
            B,
            seed,
            sampling_noise
        )


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- runtime: %s minutes ---" % round((time.time() - start_time) / 60, 2))

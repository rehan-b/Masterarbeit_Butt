import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import pandas as pd
from utils import simulate_bagging_and_variance, save_results_png, save_result_csv


def main():
    for fix in [True, False]:

        ############## Constants  ################################
        CHUNK_SIZE      = 500
        ijk             = True
        fix_x_points    = fix

        ####### Simulation parameters  ############################
        n        = 500
        n_sim    = 1_000
        B        = 10_000  # Paper uses 10_000
        args     = {"max_leaf_nodes": 5}
        seed     = 45
        new_data = np.linspace(0, 1, 250)
        ###########################################################

        bagged_preds = np.zeros((n_sim, new_data.shape[0]))
        est_vars = np.zeros((n_sim, new_data.shape[0]))

        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(
                    simulate_bagging_and_variance,
                    n=n,
                    B=B,
                    new_data=new_data,
                    simulation_index=i,
                    seed=seed,
                    dt_args=args,
                    fix_x_points=fix_x_points,
                    ijk_calculation=ijk,
                    chunk_size=CHUNK_SIZE,
                )
                for i in range(n_sim)
            ]

            for i, future in enumerate(
                tqdm(futures, desc="Simulations", unit="simulation")
            ):
                bagged_prediction, est_variance = future.result()
                bagged_preds[i, :] = bagged_prediction
                est_vars[i, :] = est_variance

        save_result_csv(
            fix_x_points=fix_x_points,
            seed=seed,
            B=B,
            args=args,
            bagged_preds=bagged_preds,
            est_vars=est_vars,
            new_data=new_data,
        )

        save_results_png(
            new_data=new_data,
            bagged_preds=bagged_preds,
            est_vars=est_vars,
            n_data_points=n,
            B=B,
            seed=seed,
            dt_args=args,
            fixed_x_points=fix_x_points,
        )


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- runtime: %s minutes ---" % round((time.time() - start_time) / 60, 2))

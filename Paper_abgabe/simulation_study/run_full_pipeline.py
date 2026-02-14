import argparse
import os
import pandas as pd

from utils import run_simulation_and_save, create_all_plots


def build_data_params(n: int, tau: float, x_pred: pd.DataFrame):
    common = {
        "shape_weibull": 1,
        "p_1": -0.405,
        "p_2": -0.4,
        "p_3": -0.05,
        "p_4": -0.01,
        "n": n,
        "tau": tau,
        "X_pred_point": x_pred,
    }

    data_generation_parameter_1 = {
        **common,
        "scale_weibull_base": 22080,
        "rate_censoring": 0.00321,
    }
    data_generation_parameter_3 = {
        **common,
        "scale_weibull_base": 18900,
        "rate_censoring": 0.01125,
    }
    data_generation_parameter_5 = {
        **common,
        "scale_weibull_base": 15120,
        "rate_censoring": 0.023293,
    }
    return data_generation_parameter_1, data_generation_parameter_3, data_generation_parameter_5


def main():
    parser = argparse.ArgumentParser(description="Run simulation + generate all plots automatically.")
    parser.add_argument("--n-sim", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=1429)
    parser.add_argument("--b-rf", type=int, default=1000)
    parser.add_argument("--b-1", type=int, default=200)
    parser.add_argument("--n-jobs", type=int, default=-1, help="-1 uses all available CPU cores")
    args = parser.parse_args()

    x_pred = pd.DataFrame([[0, 1, 80, 40]], columns=["X_1", "X_2", "X_3", "X_4"])
    data1, data3, data5 = build_data_params(n=args.n, tau=37, x_pred=x_pred)

    params_rf = {
        "n_estimators": args.b_rf,
        "max_depth": 4,
        "min_samples_split": 5,
        "max_features": "log2",
        "random_state": args.seed,
        "weighted_bootstrapping": True,
    }

    save_path = run_simulation_and_save(
        n_sim=args.n_sim,
        seed=args.seed,
        n_covariates=4,
        n=args.n,
        B_RF=args.b_rf,
        B_1=args.b_1,
        data_generation_parameter_1=data1,
        data_generation_parameter_3=data3,
        data_generation_parameter_5=data5,
        params_rf=params_rf,
        n_jobs=args.n_jobs,
    )

    create_all_plots(save_path)
    print(f"Simulation and plots completed. Output folder: {os.path.abspath(save_path)}")


if __name__ == "__main__":
    main()

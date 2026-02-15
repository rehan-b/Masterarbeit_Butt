import argparse
import os
import pandas as pd

from utils import run_simulation_and_save, create_plots_from_notebooks


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
    parser = argparse.ArgumentParser(
        description="Run simulation + generate all plots automatically.",
        epilog=(
            "PowerShell examples:\n"
            "  One-line:\n"
            "    python run_full_pipeline.py --mode plots-only --results-path \"results_in_pdf/...\" --corr-xlim-1 0.00 0.25 --corr-xlim-3 0.00 0.35 --corr-xlim-5 0.00 0.45 --strip-xlim 0.10 1.00 --rb-xlim -60 100\n"
            "\n"
            "  Multi-line in PowerShell uses backtick (`), not backslash (\\):\n"
            "    python run_full_pipeline.py `\n"
            "      --mode plots-only `\n"
            "      --results-path \"results_in_pdf/...\" `\n"
            "      --corr-xlim-1 0.00 0.25 `\n"
            "      --corr-xlim-3 0.00 0.35 `\n"
            "      --corr-xlim-5 0.00 0.45 `\n"
            "      --strip-xlim 0.10 1.00 `\n"
            "      --rb-xlim -60 100\n"
            "\n"
            "  Alternative (Windows helper script from repo root):\n"
            "    ./run_plots_only.ps1 -ResultsPath \"results_in_pdf/...\""
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--mode", choices=["simulate", "plots-only"], default="simulate")
    parser.add_argument("--results-path", type=str, default=None, help="Existing result folder for plots-only mode")
    parser.add_argument("--n-sim", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=1429)
    parser.add_argument("--b-rf", type=int, default=1000)
    parser.add_argument("--b-1", type=int, default=200)
    parser.add_argument("--n-jobs", type=int, default=-1, help="-1 uses all available CPU cores")
    parser.add_argument("--patient-label", type=str, default="averageS")
    parser.add_argument("--corr-xlim-1", nargs=2, type=float, metavar=("MIN", "MAX"), default=None)
    parser.add_argument("--corr-xlim-3", nargs=2, type=float, metavar=("MIN", "MAX"), default=None)
    parser.add_argument("--corr-xlim-5", nargs=2, type=float, metavar=("MIN", "MAX"), default=None)
    parser.add_argument("--strip-xlim", nargs=2, type=float, metavar=("MIN", "MAX"), default=None)
    parser.add_argument("--var-xlim-1", nargs=2, type=float, metavar=("MIN", "MAX"), default=None)
    parser.add_argument("--var-xlim-3", nargs=2, type=float, metavar=("MIN", "MAX"), default=None)
    parser.add_argument("--var-xlim-5", nargs=2, type=float, metavar=("MIN", "MAX"), default=None)
    parser.add_argument(
        "--rb-xlim",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        default=[-60.0, 100.0],
        help="x-axis limits for RB plot (default: -60 100)",
    )
    args = parser.parse_args()

    corr_xlims = None
    if args.corr_xlim_1 or args.corr_xlim_3 or args.corr_xlim_5:
        if not (args.corr_xlim_1 and args.corr_xlim_3 and args.corr_xlim_5):
            raise ValueError("If you set corr x-limits, provide all three: --corr-xlim-1/3/5")
        corr_xlims = [tuple(args.corr_xlim_1), tuple(args.corr_xlim_3), tuple(args.corr_xlim_5)]

    var_xlims = None
    if args.var_xlim_1 or args.var_xlim_3 or args.var_xlim_5:
        if not (args.var_xlim_1 and args.var_xlim_3 and args.var_xlim_5):
            raise ValueError("If you set variance x-limits, provide all three: --var-xlim-1/3/5")
        var_xlims = [tuple(args.var_xlim_1), tuple(args.var_xlim_3), tuple(args.var_xlim_5)]

    if args.mode == "plots-only":
        if args.results_path is None:
            raise ValueError("For --mode plots-only you must provide --results-path")
        create_plots_from_notebooks(
            args.results_path,
            patient=args.patient_label,
            corr_xlims=corr_xlims,
            strip_xlim=tuple(args.strip_xlim) if args.strip_xlim else None,
            var_xlims=var_xlims,
            rb_xlim=tuple(args.rb_xlim),
        )
        print(f"Notebook plots generated for existing folder: {os.path.abspath(args.results_path)}")
        return

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

    create_plots_from_notebooks(
        save_path,
        patient=args.patient_label,
        corr_xlims=corr_xlims,
        strip_xlim=tuple(args.strip_xlim) if args.strip_xlim else None,
        var_xlims=var_xlims,
        rb_xlim=tuple(args.rb_xlim),
    )
    print(f"Simulation and plots completed. Output folder: {os.path.abspath(save_path)}")


if __name__ == "__main__":
    main()

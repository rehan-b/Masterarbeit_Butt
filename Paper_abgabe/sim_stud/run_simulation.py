"""
Unified simulation pipeline — kontinuierliche und binäre Kovariaten.

════════════════════════════════════════════════════════════════════
 BEISPIELBEFEHLE
════════════════════════════════════════════════════════════════════

── Kontinuierliche Kovariaten (X_3 ~ N(80,10), X_4 ~ N(40,5)) ────

  Scenario 1 (~20% Events):
    python run_simulation.py --study original --scenario 1 --n-sim 1000 --seed 42 --n 1429 --b-rf 1000 --b-1 200 --n-jobs -1

  Scenario 2 (~40% Events):
    python run_simulation.py --study original --scenario 2 --n-sim 1000 --seed 42 --n 1429 --b-rf 1000 --b-1 200 --n-jobs -1

── Binäre Kovariaten (X_3 ~ B(0.9), X_4 ~ B(0.4)) ────────────────

  Scenario 1 (~20% Events):
    python run_simulation.py --study binary --scenario 1 --n-sim 1000 --seed 42 --n 1429 --b-rf 1000 --b-1 200 --n-jobs -1

  Scenario 2 (~40% Events):
    python run_simulation.py --study binary --scenario 2 --n-sim 1000 --seed 42 --n 1429 --b-rf 1000 --b-1 200 --n-jobs -1

── Nur Plots neu erzeugen (ohne Simulation) ───────────────────────

    python run_simulation.py --mode plots-only --results-path "rev2_res/(n_train)1000__(B_RF)1000__.../__index__"

  Mit Achsengrenzen (PowerShell, backtick für Zeilenumbruch):
    python run_simulation.py `
      --mode plots-only `
      --results-path "rev2_res/..." `
      --corr-xlim-1 0.00 0.25 `
      --corr-xlim-3 0.00 0.35 `
      --corr-xlim-5 0.00 0.45 `
      --strip-xlim 0.10 1.00 `
      --rb-xlim -60 100

── Ausgabeverzeichnisse ────────────────────────────────────────────

  original / scenario 1  →  rev2_res/(n_train)...4kovariates/
  original / scenario 2  →  rev2_res/(n_train)...4kovariates_events40/
  binary   / scenario 1  →  res_4binary/(n_train)...4kovariates_binary_events20/
  binary   / scenario 2  →  res_4binary/(n_train)...4kovariates_binary_events40/

════════════════════════════════════════════════════════════════════
"""

import argparse
import os
import pandas as pd

from utils import run_simulation_and_save, create_plots_from_notebooks


# ── Parameter: kontinuierliche Kovariaten ─────────────────────────────────────
# X_1~B(0.3), X_2~B(0.8), X_3~N(80,10), X_4~N(40,5)
# p_1=-0.405, p_2=-0.4, p_3=-0.05, p_4=-0.01  |  X_pred=[0,1,80,40]

PARAMS_ORIGINAL_S1 = {          # Scenario 1: ~20% Events vor τ
    "1": {"scale_weibull_base": 22080, "rate_censoring": 0.00321  },  # ~10% zero-weights
    "3": {"scale_weibull_base": 18900, "rate_censoring": 0.01125  },  # ~30% zero-weights
    "5": {"scale_weibull_base": 15120, "rate_censoring": 0.023293 },  # ~50% zero-weights
}

PARAMS_ORIGINAL_S2 = {          # Scenario 2: ~40% Events vor τ
    "1": {"scale_weibull_base":  8951, "rate_censoring": 0.00378  },  # ~10% zero-weights
    "3": {"scale_weibull_base":  7100, "rate_censoring": 0.01420  },  # ~30% zero-weights
    "5": {"scale_weibull_base":  4665, "rate_censoring": 0.03550  },  # ~50% zero-weights
}

# ── Parameter: binäre Kovariaten ──────────────────────────────────────────────
# X_1~B(0.3), X_2~B(0.8), X_3~B(0.9), X_4~B(0.4)
# p_1=-0.405, p_2=-0.4, p_3=-0.35, p_4=-0.30  |  X_pred=[0,1,1,0]

PARAMS_BINARY_S1 = {            # Scenario 1: ~20% Events vor τ
    "1": {"scale_weibull_base": 369.01, "rate_censoring": 0.003059 },  # ~10% zero-weights
    "3": {"scale_weibull_base": 332.93, "rate_censoring": 0.011002 },  # ~30% zero-weights
    "5": {"scale_weibull_base": 281.63, "rate_censoring": 0.022153 },  # ~50% zero-weights
}

PARAMS_BINARY_S2 = {            # Scenario 2: ~40% Events vor τ
    "1": {"scale_weibull_base": 167.30, "rate_censoring": 0.003308 },  # ~10% zero-weights
    "3": {"scale_weibull_base": 131.44, "rate_censoring": 0.013171 },  # ~30% zero-weights
    "5": {"scale_weibull_base":  86.26, "rate_censoring": 0.032175 },  # ~50% zero-weights
}

# Lookup: (study, scenario) → parameter dict
_PARAMS = {
    ("original", "1"): PARAMS_ORIGINAL_S1,
    ("original", "2"): PARAMS_ORIGINAL_S2,
    ("binary",   "1"): PARAMS_BINARY_S1,
    ("binary",   "2"): PARAMS_BINARY_S2,
}

# Lookup: (study, scenario) → (results_subdir, exp_name_suffix)
_OUTPUT = {
    ("original", "1"): ("rev2_res",   ""               ),
    ("original", "2"): ("rev2_res",   "_events40"      ),
    ("binary",   "1"): ("res_4binary", "_binary_events20"),
    ("binary",   "2"): ("res_4binary", "_binary_events40"),
}


def build_data_params(n: int, tau: float, study: str, scenario: str):
    pw = _PARAMS[(study, scenario)]

    if study == "original":
        common = {
            "shape_weibull": 1,
            "p_1": -0.405, "p_2": -0.4, "p_3": -0.05, "p_4": -0.01,
            "n": n, "tau": tau,
            "X_pred_point": pd.DataFrame([[0, 1, 80, 40]], columns=["X_1","X_2","X_3","X_4"]),
        }
    else:
        common = {
            "shape_weibull": 1,
            "p_1": -0.405, "p_2": -0.4, "p_3": -0.35, "p_4": -0.30,
            "n": n, "tau": tau,
            "X_pred_point": pd.DataFrame([[0, 1, 1, 0]], columns=["X_1","X_2","X_3","X_4"]),
            "covariate_type": "binary",
        }

    return (
        {**common, **pw["1"]},
        {**common, **pw["3"]},
        {**common, **pw["5"]},
    )


def main():
    parser = argparse.ArgumentParser(
        description="Unified simulation pipeline (original + binary covariates).",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--study", choices=["original", "binary"], default="original",
        help=(
            "original: X_3~N(80,10), X_4~N(40,5)  →  rev2_res/\n"
            "binary:   X_3~B(0.9),   X_4~B(0.4)   →  res_4binary/"
        ),
    )
    parser.add_argument(
        "--scenario", choices=["1", "2"], default="1",
        help="1 = ~20%% Events vor τ  |  2 = ~40%% Events vor τ",
    )
    parser.add_argument("--mode", choices=["simulate", "plots-only"], default="simulate")
    parser.add_argument("--results-path", type=str, default=None,
                        help="Vorhandener Ergebnisordner (nur für --mode plots-only)")
    parser.add_argument("--n-sim",  type=int, default=1000)
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--n",      type=int, default=1429)
    parser.add_argument("--b-rf",   type=int, default=1000)
    parser.add_argument("--b-1",    type=int, default=200)
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="-1 = alle verfügbaren CPU-Kerne")
    parser.add_argument("--patient-label", type=str, default="averageS")
    parser.add_argument("--corr-xlim-1", nargs=2, type=float, metavar=("MIN","MAX"), default=None)
    parser.add_argument("--corr-xlim-3", nargs=2, type=float, metavar=("MIN","MAX"), default=None)
    parser.add_argument("--corr-xlim-5", nargs=2, type=float, metavar=("MIN","MAX"), default=None)
    parser.add_argument("--strip-xlim",  nargs=2, type=float, metavar=("MIN","MAX"), default=None)
    parser.add_argument("--var-xlim-1",  nargs=2, type=float, metavar=("MIN","MAX"), default=None)
    parser.add_argument("--var-xlim-3",  nargs=2, type=float, metavar=("MIN","MAX"), default=None)
    parser.add_argument("--var-xlim-5",  nargs=2, type=float, metavar=("MIN","MAX"), default=None)
    parser.add_argument("--rb-xlim", nargs=2, type=float, metavar=("MIN","MAX"),
                        default=[-60.0, 100.0],
                        help="x-Achse RB-Plot (default: -60 100)")
    args = parser.parse_args()

    # ── Achsengrenzen aufbereiten ──────────────────────────────────────────────
    corr_xlims = None
    if args.corr_xlim_1 or args.corr_xlim_3 or args.corr_xlim_5:
        if not (args.corr_xlim_1 and args.corr_xlim_3 and args.corr_xlim_5):
            raise ValueError("Wenn corr-xlim gesetzt, alle drei angeben: --corr-xlim-1/3/5")
        corr_xlims = [tuple(args.corr_xlim_1), tuple(args.corr_xlim_3), tuple(args.corr_xlim_5)]

    var_xlims = None
    if args.var_xlim_1 or args.var_xlim_3 or args.var_xlim_5:
        if not (args.var_xlim_1 and args.var_xlim_3 and args.var_xlim_5):
            raise ValueError("Wenn var-xlim gesetzt, alle drei angeben: --var-xlim-1/3/5")
        var_xlims = [tuple(args.var_xlim_1), tuple(args.var_xlim_3), tuple(args.var_xlim_5)]

    plot_kwargs = dict(
        patient=args.patient_label,
        corr_xlims=corr_xlims,
        strip_xlim=tuple(args.strip_xlim) if args.strip_xlim else None,
        var_xlims=var_xlims,
        rb_xlim=tuple(args.rb_xlim),
    )

    # ── plots-only ────────────────────────────────────────────────────────────
    if args.mode == "plots-only":
        if args.results_path is None:
            raise ValueError("--mode plots-only benötigt --results-path")
        create_plots_from_notebooks(args.results_path, **plot_kwargs)
        print(f"Plots erzeugt: {os.path.abspath(args.results_path)}")
        return

    # ── simulate ──────────────────────────────────────────────────────────────
    data1, data3, data5 = build_data_params(
        n=args.n, tau=37, study=args.study, scenario=args.scenario
    )

    params_rf = {
        "n_estimators": args.b_rf,
        "max_depth": 4,
        "min_samples_split": 5,
        "max_features": "log2",
        "random_state": args.seed,
        "weighted_bootstrapping": True,
    }

    results_subdir, exp_name_suffix = _OUTPUT[(args.study, args.scenario)]

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
        results_subdir=results_subdir,
        exp_name_suffix=exp_name_suffix,
    )

    create_plots_from_notebooks(save_path, **plot_kwargs)
    print(f"Simulation + Plots abgeschlossen. Ordner: {os.path.abspath(save_path)}")


if __name__ == "__main__":
    main()

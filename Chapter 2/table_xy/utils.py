import numpy as np
import pandas as pd

def create_bootstrap_indices_and_Nbi(
    n: int, B: int, seed: int = None, weights: np.ndarray = None
):
    if weights is None:
        rng = np.random.default_rng(seed)
        boot_indices = rng.choice(np.arange(n), size=(B, n), replace=True)
        boot_counts = np.apply_along_axis(
            lambda x: np.bincount(x, minlength=n), axis=1, arr=boot_indices
        )
        return boot_indices, boot_counts

    else:
        rng = np.random.default_rng(seed)
        boot_indices = rng.choice(np.arange(n), size=(B, n), p=weights, replace=True)
        boot_counts = np.apply_along_axis(
            lambda x: np.bincount(x, minlength=n), axis=1, arr=boot_indices
        )
        return boot_indices, boot_counts


def bagging_mean_estimators(x, B, seed, weights):
    n = x.shape[0]
    T_N_b = np.zeros(B)
    indices_list, N_bi = create_bootstrap_indices_and_Nbi(
        n=n, B=B, seed=seed, weights=weights
    )

    for b in range(B):
        indices = indices_list[b]
        T_N_b[b] = np.mean(x[indices])

    return T_N_b, N_bi


def inf_JK_bagged_variance_weighted(N_bi, T_N_b, weights,m) :
    B, n = N_bi.shape
    T_N_b_mean = np.mean(T_N_b, axis=0)
    

    cov_i = ((N_bi - n * weights).T @ (T_N_b - T_N_b_mean)) / B
    cov_i_hoch2 = cov_i**2
    array = cov_i_hoch2/weights

    biased_var_estimate = np.sum(array[~np.isnan(array) & ~np.isinf(array)], axis=0) * np.sum(weights**2)

    bias_correction = n / B * np.sum(1-weights[weights > 0]) * np.var(T_N_b, axis=0, ddof=1)* np.sum(weights**2)

    return biased_var_estimate, bias_correction


def simulate_bagging_and_ijk_var_calculation(x1, B, seed, sim_i, weights, m):
    T_N_b, N_bi = bagging_mean_estimators(x=x1, B=B, seed=seed + sim_i, weights=weights)
    biased_var_estimate, bias_correction = inf_JK_bagged_variance_weighted(
        N_bi=N_bi, T_N_b=T_N_b, weights=weights, m=m
    )

    ijk_var_bagged_est = biased_var_estimate - bias_correction
    theta_bagged = T_N_b.mean()
    return biased_var_estimate,bias_correction, theta_bagged



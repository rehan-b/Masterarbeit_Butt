import numpy as np

def create_bootstrap_indices_and_Nbi(n: int, B: int, seed: int = None, weights: np.ndarray = None):
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


def bagging_mean_estimators(x,B,seed, weights):
    n = x.shape[0]
    T_N_b = np.zeros(B)
    indices_list, N_bi = create_bootstrap_indices_and_Nbi(n=n, B=B, seed=seed, weights=weights)
    
    for b in range(B):
        indices = indices_list[b]
        T_N_b[b] = np.mean(x[indices])

    return T_N_b, N_bi


def inf_JK_bagged_variance(N_bi: np.ndarray, T_N_b: np.ndarray) :
    B, n = N_bi.shape
    T_N_b_mean = np.mean(T_N_b, axis=0)

    cov_i = ((N_bi - 1).T @ (T_N_b - T_N_b_mean)) / B
    cov_i_hoch2 = cov_i**2
    biased_var_estimate = np.sum(cov_i_hoch2, axis=0)
    
    bias_correction = ((n - 1)/ B) * np.var(T_N_b, axis=0)
    return biased_var_estimate, bias_correction


def inf_JK_bagged_variance_weighted(N_bi, T_N_b, weights, m):
    B, n = N_bi.shape
    T_N_b_mean = np.mean(T_N_b, axis=0)

    cov_i = ((N_bi - n * weights[0]).T @ (T_N_b- T_N_b_mean)) / B 
    cov_i_hoch2 = cov_i**2
    biased_var_estimate = np.sum(cov_i_hoch2)
    
    bias_correction = n/B * (m-1)/m * np.var(T_N_b)
    
    return biased_var_estimate, bias_correction


def simulate_bagging_and_ijk_var_calculation(x1, B, seed, sim_i, weights, m ) :
    T_N_b, N_bi = bagging_mean_estimators(x=x1, B=B, seed=seed+sim_i, weights=weights)
    biased_var_estimate, bias_correction =inf_JK_bagged_variance_weighted(N_bi=N_bi, T_N_b=T_N_b, weights=weights, m=m)

    ijk_var_bagged_est = biased_var_estimate - bias_correction
    theta_bagged = T_N_b.mean()
    return ijk_var_bagged_est, theta_bagged



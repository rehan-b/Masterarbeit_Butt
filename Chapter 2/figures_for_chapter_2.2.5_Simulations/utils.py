import numpy as np


def simulate_mean(i, n, B, seed):
    np.random.seed(seed + i)
    x = np.random.normal(0, 1, n)

    # Jackknife variance
    var_jackknife = np.var(x, ddof=1) / n

    # Bootstrap variance
    mean = np.array([np.mean(np.random.choice(x, n, replace=True)) for _ in range(B)])
    var_boot = np.var(mean, ddof=1)

    var_ijk = (np.var(x, ddof=1) / n) * ((n - 1) / n)

    return var_jackknife, var_boot, var_ijk


################################################################################################


def jackknife_corr(x, y, func):
    n = len(x)
    idx = np.arange(n)
    jack_i = [func(x[idx != i], y[idx != i])[0, 1] for i in range(n)]
    jack_mean = np.mean(jack_i)
    return ((n - 1) / n) * np.sum((jack_i - jack_mean) ** 2)


def bootstrap_corr(x, y, func, B):
    n = len(x)
    idxs = np.random.choice(np.arange(n), (B, n), replace=True)
    bootstrap = [func(x[idx], y[idx])[0, 1] for idx in idxs]
    return np.var(bootstrap, ddof=1)


def mean_weighted(x, p_i):
    return np.sum(x * p_i)


def inf_jack_corr(x, y, func, e):
    n = len(x)
    T_0 = func(x, y)[0, 1]
    U_i = np.zeros(len(x))
    for i in range(n):
        weights_inf_jk = np.full(n, (1 - e) / n)
        weights_inf_jk[i] += e
        mean_x = mean_weighted(x, weights_inf_jk)
        mean_y = mean_weighted(y, weights_inf_jk)
        T_weighted = np.sum(weights_inf_jk * (x - mean_x) * (y - mean_y)) / np.sqrt(
            np.sum(weights_inf_jk * (x - mean_x) ** 2)
            * np.sum(weights_inf_jk * (y - mean_y) ** 2)
        )
        U_i[i] = (T_weighted - T_0) / e
    return np.sum(U_i**2) / n**2


def simulate_pearson(mean, cov, n, B, sim_i, seed):
    np.random.seed(seed + sim_i)
    x, y = np.random.multivariate_normal(mean, cov, n).T
    var_jackknife = jackknife_corr(x, y, np.corrcoef)
    var_boot = bootstrap_corr(x, y, np.corrcoef, B)
    var_ijk = inf_jack_corr(x, y, np.corrcoef, 0.000000001)
    return var_jackknife, var_boot, var_ijk


################################################################################################


def jackknife_median(x, func):
    n = len(x)
    idx = np.arange(n)
    jack_i = [func(x[idx != i]) for i in range(n)]
    jack_mean = np.mean(jack_i)
    return ((n - 1) / n) * np.sum((jack_i - jack_mean) ** 2)


def bootstrap_median(x, func, B):
    n = len(x)
    idxs = np.random.choice(np.arange(n), (B, n), replace=True)
    bootstrap = [func(x[idx]) for idx in idxs]
    return np.var(bootstrap, ddof=1)


def weighted_quantile(
    values, quantiles, sample_weight=None, values_sorted=False, old_style=False
):
    """Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(
        quantiles <= 1
    ), "quantiles should be in [0, 1]"

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)


def inf_jack_median(x, func, e):
    n = len(x)
    T_0 = func(x)
    U_i = np.zeros(n)
    for i in range(n):
        weights_inf_jk = np.full(n, (1 - e) / n)
        weights_inf_jk[i] += e
        T_weighted = weighted_quantile(x, 0.5, sample_weight=weights_inf_jk * 200)
        U_i[i] = (T_weighted - T_0) / e
    return np.sum(U_i**2) / n**2


def simulate_median(n, B, sim_i, seed):
    np.random.seed(seed + sim_i)
    x = np.random.normal(0, 1, n)
    var_jackknife = jackknife_median(x=x, func=np.median)
    var_boot = bootstrap_median(x=x, func=np.median, B=B)
    var_emp = np.median(x)
    var_inf = inf_jack_median(x, np.median, 0.000000001)
    return var_jackknife, var_boot, var_emp, var_inf

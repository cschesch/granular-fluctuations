#!/usr/bin/env python3
"""Factor-rank specification tests on the grouped covariance (Section "How many common factors?").

For each candidate rank r, fit a Gaussian factor-analysis covariance, compute the
normal-theory LR statistic and the asymptotically distribution-free (ADF/GMM) J
statistic, then turn the sequence of rejections into a one-sided lower bound on
the number of common factors. Used by ``check_factor_rank_figure1.py``; requires
jax + jaxopt.
"""

from __future__ import annotations


import numpy as np
import pandas as pd
from scipy.stats import chi2

from mgp_core import sample_cov


# Reuse JAX solvers across panels with the same numerical configuration to avoid
# recompiling the objectives.
_JAX_RANK_SOLVERS: dict[tuple, tuple] = {}


def _factor_rank_jax_solvers(
    n: int,
    rank: int,
    maxiter_ml: int,
    maxiter_gmm: int,
    tol_ml: float,
    tol_gmm: float,
):
    key = (n, rank, maxiter_ml, maxiter_gmm, float(tol_ml), float(tol_gmm))
    cached = _JAX_RANK_SOLVERS.get(key)
    if cached is not None:
        return cached

    import jax.numpy as jnp
    import jaxopt

    def unpack(theta):
        loading = theta[: n * rank].reshape((n, rank))
        psi = jnp.exp(theta[n * rank:])
        return loading, psi

    def lr_obj(theta, sigma_j, logdet_hat_j, t_j):
        loading, psi = unpack(theta)
        sigma_r = loading @ loading.T + jnp.diag(psi)
        _, logdet_r = jnp.linalg.slogdet(sigma_r)
        trace_term = jnp.trace(jnp.linalg.solve(sigma_r, sigma_j))
        return t_j * (logdet_r - logdet_hat_j + trace_term - n)

    def gmm_obj(theta, sigma_j, inv_omega_j, tri_i, tri_j, t_j):
        loading, psi = unpack(theta)
        sigma_r = loading @ loading.T + jnp.diag(psi)
        residual = sigma_j - sigma_r
        moments = residual[tri_i, tri_j]
        return t_j * (moments @ (inv_omega_j @ moments))

    cached = (
        jaxopt.BFGS(fun=lr_obj, maxiter=maxiter_ml, tol=tol_ml, jit=True),
        jaxopt.BFGS(fun=gmm_obj, maxiter=maxiter_gmm, tol=tol_gmm, jit=True),
        gmm_obj,
    )
    _JAX_RANK_SOLVERS[key] = cached
    return cached



# The distribution-free J statistic inverts the M(M+1)/2-dimensional empirical
# covariance of the sample second moments.  Its ordinary inverse cannot exist
# unless T-1 is at least that dimension.  We also retain the conservative M<=10
# reporting ceiling used in the paper.
MAX_M_FOR_J_TEST = 10


def j_test_reportable(m: int, t: int | None = None) -> bool:
    """Whether the ordinary distribution-free J weight is estimable."""
    m = int(m)
    enough_observations = t is None or int(t) - 1 >= m * (m + 1) // 2
    return m <= MAX_M_FOR_J_TEST and enough_observations


def factor_rank_ledermann(n: int) -> int:
    """Ledermann bound: largest rank whose factor model has df >= 0."""
    return int(n + 0.5 * (1.0 - np.sqrt(8 * n + 1)))


def factor_rank_max_testable(n: int) -> int:
    """Largest rank with a non-degenerate chi-square test, i.e. df > 0.

    This is the Ledermann bound itself whenever df is strictly positive there,
    and one rank below it when the bound is exactly saturated (df = 0, as at
    N = 10 and N = 45).  df falls by n - r as r increases, so it is strictly
    decreasing over the admissible range and the search below is well posed.
    """
    ledermann = factor_rank_ledermann(n)
    return ledermann if factor_rank_degrees_of_freedom(n, ledermann) > 0 else ledermann - 1


def factor_rank_degrees_of_freedom(n: int, r: int) -> int:
    """Chi-square degrees of freedom for an N-variable rank-r factor-analysis model."""
    p_full = n * (n + 1) // 2
    p_r = n * r + n - r * (r - 1) // 2
    return int(p_full - p_r)


def _regularize_spd(matrix: np.ndarray, rel_jitter: float = 1e-8) -> tuple[np.ndarray, float]:
    matrix = 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)
    n = matrix.shape[0]
    scale = float(np.trace(matrix) / max(n, 1))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.linalg.norm(matrix)) or 1.0
    eig_min = float(np.linalg.eigvalsh(matrix).min())
    target = rel_jitter * scale
    jitter = max(0.0, target - eig_min)
    if jitter > 0:
        matrix = matrix + jitter * np.eye(n)
    return matrix, float(jitter)


def _factor_rank_initial_theta(sigma: np.ndarray, r: int) -> np.ndarray:
    evals, evecs = np.linalg.eigh(0.5 * (sigma + sigma.T))
    order = np.argsort(evals)[::-1][:r]
    evals = evals[order]
    evecs = evecs[:, order]
    loading = evecs * np.sqrt(np.maximum(evals, 1e-10))
    psi_floor = max(float(np.mean(np.diag(sigma))) * 1e-8, 1e-12)
    psi = np.maximum(np.diag(sigma) - np.sum(loading**2, axis=1), psi_floor)
    return np.concatenate([loading.ravel(), np.log(psi)])


def _factor_rank_cov_from_theta(theta: np.ndarray, n: int, r: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loading = np.asarray(theta[: n * r], dtype=float).reshape(n, r)
    psi = np.exp(np.asarray(theta[n * r:], dtype=float))
    sigma_r = loading @ loading.T + np.diag(psi)
    return loading, psi, 0.5 * (sigma_r + sigma_r.T)


def _factor_rank_lr_from_theta(theta: np.ndarray, sigma: np.ndarray, n: int, r: int, t: int, logdet_hat: float) -> float:
    _, _, sigma_r = _factor_rank_cov_from_theta(theta, n, r)
    sign, logdet_r = np.linalg.slogdet(sigma_r)
    if sign <= 0:
        return np.nan
    trace_term = float(np.trace(np.linalg.solve(sigma_r, sigma)))
    return float(t * (logdet_r - logdet_hat + trace_term - n))


def _factor_rank_sklearn_theta(
    centered_x: np.ndarray,
    sigma: np.ndarray,
    r: int,
    maxiter: int,
    tol: float,
) -> tuple[np.ndarray, str]:
    if r == 0:
        psi_floor = max(float(np.mean(np.diag(sigma))) * 1e-8, 1e-12)
        psi = np.maximum(np.diag(sigma), psi_floor)
        return np.log(psi), "sklearn_not_needed"
    try:
        import warnings

        from sklearn.decomposition import FactorAnalysis
        from sklearn.exceptions import ConvergenceWarning

        fa = FactorAnalysis(
            n_components=r,
            tol=tol,
            max_iter=maxiter,
            svd_method="lapack",
            rotation=None,
            random_state=0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            fa.fit(centered_x)
        loading = np.asarray(fa.components_, dtype=float).T
        psi_floor = max(float(np.mean(np.diag(sigma))) * 1e-8, 1e-12)
        psi = np.maximum(np.asarray(fa.noise_variance_, dtype=float), psi_floor)
        return np.concatenate([loading.ravel(), np.log(psi)]), "sklearn"
    except Exception:
        return _factor_rank_initial_theta(sigma, r), "spectral_fallback"


def _factor_rank_gmm_weight(
    centered_x: np.ndarray,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    n = centered_x.shape[1]
    tri = np.triu_indices(n)
    moments = centered_x[:, tri[0]] * centered_x[:, tri[1]]
    omega = np.cov(moments, rowvar=False, ddof=1)
    q = len(tri[0])
    if centered_x.shape[0] - 1 < q:
        raise ValueError(
            f"ADF J weight is not estimable: T-1={centered_x.shape[0]-1} < "
            f"M(M+1)/2={q}."
        )
    rank = int(np.linalg.matrix_rank(omega))
    if rank < q:
        raise np.linalg.LinAlgError(
            f"ADF J moment covariance is rank deficient: rank={rank} < {q}."
        )
    # This is deliberately the ordinary inverse described in the paper, not a
    # pseudoinverse or ridge repair.
    inv_omega = np.linalg.inv(omega)
    return inv_omega, tri


def factor_rank_tests(
    x: np.ndarray,
    alpha: float = 0.05,
    rel_jitter: float = 1e-8,
    maxiter_ml: int = 2000,
    maxiter_gmm: int = 2000,
    tol_ml: float = 1e-6,
    tol_gmm: float = 1e-7,
    progress_label: str | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Test every rank from zero through the largest positive-df model.

    Fit each covariance by Gaussian maximum likelihood, then minimize the
    distribution-free GMM objective from that fit. Return rank-level diagnostic
    statistics and the first non-rejected rank, using the Ledermann bound when
    every testable rank is rejected.
    """
    try:
        import jax

        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("factor_rank_tests requires jax and jaxopt in the active Python environment.") from exc

    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"factor_rank_tests expects a T x N array, got shape {x.shape}.")
    if x.shape[0] < 3 or x.shape[1] < 3:
        raise ValueError(f"factor_rank_tests needs at least T>=3 and N>=3, got {x.shape}.")
    if not np.isfinite(x).all():
        raise ValueError("factor_rank_tests input contains non-finite values.")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0,1), got {alpha}.")

    t, n = x.shape
    if not j_test_reportable(n, t):
        q = n * (n + 1) // 2
        raise ValueError(
            f"Distribution-free J test is not reportable at M={n}, T={t}: "
            f"it requires M<=10 and T-1 >= M(M+1)/2={q}."
        )
    centered = x - x.mean(axis=0, keepdims=True)
    sigma_raw = sample_cov(centered)
    sigma, sigma_jitter = _regularize_spd(sigma_raw, rel_jitter=rel_jitter)
    sign, logdet_hat = np.linalg.slogdet(sigma)
    if sign <= 0:
        raise RuntimeError("Regularized sample covariance is not positive definite.")

    ledermann = factor_rank_ledermann(n)
    upper = factor_rank_max_testable(n)
    ranks = range(upper + 1)
    inv_omega, tri = _factor_rank_gmm_weight(centered)
    omega_condition = float(np.linalg.cond(inv_omega))

    sigma_j = jnp.asarray(sigma)
    logdet_hat_j = jnp.asarray(logdet_hat)
    inv_omega_j = jnp.asarray(inv_omega)
    tri_i = jnp.asarray(tri[0])
    tri_j = jnp.asarray(tri[1])

    rows: list[dict[str, object]] = []
    for rank in ranks:
        df = factor_rank_degrees_of_freedom(n, rank)
        theta_sklearn_np, init_status = _factor_rank_sklearn_theta(
            centered,
            sigma,
            rank,
            maxiter=maxiter_ml,
            tol=tol_ml,
        )
        lr_sklearn = _factor_rank_lr_from_theta(theta_sklearn_np, sigma, n, rank, t, logdet_hat)
        theta0 = jnp.asarray(theta_sklearn_np)
        ml_solver, gmm_solver, gmm_obj = _factor_rank_jax_solvers(
            n, rank, maxiter_ml, maxiter_gmm, tol_ml, tol_gmm
        )
        t_j = jnp.asarray(float(t))

        ml_status = "ok"
        try:
            sol_ml = ml_solver.run(theta0, sigma_j, logdet_hat_j, t_j)
            theta_ml = sol_ml.params
            lr_stat = float(sol_ml.state.value)
            ml_error = float(getattr(sol_ml.state, "error", np.nan))
        except Exception as exc:  # pragma: no cover - optimizer dependent
            theta_ml = theta0
            lr_stat = np.nan
            ml_error = np.nan
            ml_status = f"{type(exc).__name__}: {exc}"
        if (
            not np.isfinite(lr_stat)
            or lr_stat < -1e-6
            or (np.isfinite(lr_sklearn) and lr_stat > lr_sklearn + max(1e-6, abs(lr_sklearn) * 1e-5))
        ):
            theta_ml = jnp.asarray(theta_sklearn_np)
            lr_stat = max(float(lr_sklearn), 0.0) if np.isfinite(lr_sklearn) else np.nan
            ml_status = f"{init_status}_fallback"
        elif lr_stat < 0:
            lr_stat = 0.0

        j_stat = np.nan
        j_ml = np.nan
        gmm_error = np.nan
        gmm_status = "skipped"
        if np.isfinite(lr_stat):
            gmm_status = "ok"
            try:
                j_ml = float(gmm_obj(theta_ml, sigma_j, inv_omega_j, tri_i, tri_j, t_j))
            except Exception:
                j_ml = np.nan
            try:
                sol_gmm = gmm_solver.run(theta_ml, sigma_j, inv_omega_j, tri_i, tri_j, t_j)
                j_stat = float(sol_gmm.state.value)
                gmm_error = float(getattr(sol_gmm.state, "error", np.nan))
            except Exception as exc:  # pragma: no cover - optimizer dependent
                gmm_status = f"{type(exc).__name__}: {exc}"
                j_stat = j_ml
            if (
                (not np.isfinite(j_stat))
                or j_stat < -1e-6
                or (np.isfinite(j_ml) and j_stat > j_ml + max(1e-6, abs(j_ml) * 1e-5))
            ):
                j_stat = max(float(j_ml), 0.0) if np.isfinite(j_ml) else np.nan
                gmm_status = "ml_moment_fallback"
            elif j_stat < 0:
                j_stat = 0.0

        crit = float(chi2.ppf(1.0 - alpha, df))
        lr_for_p = max(lr_stat, 0.0) if np.isfinite(lr_stat) else np.nan
        j_for_p = max(j_stat, 0.0) if np.isfinite(j_stat) else np.nan
        row = {
            "r": int(rank),
            "df": int(df),
            "LR": float(lr_stat),
            "J": float(j_stat),
            "crit": crit,
            "pvalue_LR": float(1.0 - chi2.cdf(lr_for_p, df)) if np.isfinite(lr_for_p) else np.nan,
            "pvalue_J": float(1.0 - chi2.cdf(j_for_p, df)) if np.isfinite(j_for_p) else np.nan,
            "reject_LR": bool(lr_stat > crit) if np.isfinite(lr_stat) else False,
            "reject_J": bool(j_stat > crit) if np.isfinite(j_stat) else False,
            "ml_error": ml_error,
            "gmm_error": gmm_error,
            "ml_status": ml_status,
            "gmm_status": gmm_status,
        }
        rows.append(row)
        if progress_label is not None:
            print(
                f"[{progress_label}] rank {rank}: LR={lr_stat:.4g}, "
                f"J={j_stat:.4g}, crit={crit:.4g}",
                flush=True,
            )
    table = pd.DataFrame(rows).sort_values("r").reset_index(drop=True)
    if table.empty:
        raise RuntimeError("No factor-rank test rows were computed.")

    def first_accepted(stat_col: str) -> int:
        vals = table[stat_col].to_numpy(dtype=float)
        crit_vals = table["crit"].to_numpy(dtype=float)
        ok = np.isfinite(vals) & (vals <= crit_vals)
        if ok.any():
            return int(table.loc[ok, "r"].iloc[0])
        return int(table["r"].iloc[-1])

    def lower_bound(stat_col: str) -> int:
        # Sequential-test lower bound on the number of common factors: reject
        # H_0(k) (a k-factor covariance fits) => rank(C) > k. The lower bound is the
        # smallest rank NOT rejected; if every testable (df>0) rank is rejected, the
        # data support at least the Ledermann bound k_bar(M) many factors.
        vals = table[stat_col].to_numpy(dtype=float)
        crit_vals = table["crit"].to_numpy(dtype=float)
        accepted = np.isfinite(vals) & (vals <= crit_vals)
        if accepted.any():
            return int(table.loc[accepted, "r"].iloc[0])
        return int(ledermann)

    def all_rejected(stat_col: str) -> bool:
        vals = table[stat_col].to_numpy(dtype=float)
        crit_vals = table["crit"].to_numpy(dtype=float)
        return not bool((np.isfinite(vals) & (vals <= crit_vals)).any())

    r_lr = first_accepted("LR")
    r_j = first_accepted("J") if table["J"].notna().any() else np.nan
    has_j = bool(table["J"].notna().any())
    lb_lr = lower_bound("LR")
    lb_j = lower_bound("J") if has_j else np.nan
    kbar = int(ledermann)
    summary = {
        "T": int(t),
        "N_rank_test": int(n),
        "alpha": float(alpha),
        "min_rank": int(table["r"].min()),
        "max_rank": int(table["r"].max()),
        "ledermann_rank_grid_stop": int(ledermann),
        "ledermann_kbar": kbar,
        "selected_rank_LR": int(r_lr),
        "selected_rank_J": int(r_j) if pd.notna(r_j) else np.nan,
        # Distribution-free (ADF/J) and normal-theory (LR) lower bounds on r=rank(C),
        # and whether the J bound already certifies the regular regime r>=k_bar(M).
        "factor_lb_LR": int(lb_lr),
        "factor_lb_J": int(lb_j) if pd.notna(lb_j) else np.nan,
        "regular_certified_J": bool(pd.notna(lb_j) and lb_j >= kbar),
        # True when no rank in the grid is accepted, so the lower bound is the
        # fallback k_bar rather than an accepted rank.
        "all_ranks_rejected_J": bool(has_j and all_rejected("J")),
        "max_testable_rank": int(table["r"].max()),
        "sigma_jitter": sigma_jitter,
        "moment_cov_dimension": int(n * (n + 1) // 2),
        "moment_cov_rank": int(n * (n + 1) // 2),
        "moment_cov_condition": omega_condition,
    }
    return table, summary

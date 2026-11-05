#!/usr/bin/env python3
"""Factor-rank specification tests on the grouped covariance (Section "How many common factors?").

For each candidate rank r, fit a Gaussian factor-analysis covariance, compute the
normal-theory LR statistic, the asymptotically distribution-free (ADF/GMM) J
statistic, and the Hayakawa (2019) mean-and-variance-corrected threshold, then
turn the sequence of rejections into a one-sided lower bound on the number of
common factors. Consumed by ``check_factor_rank.py``; requires jax + jaxopt.
"""

from __future__ import annotations


import numpy as np
import pandas as pd
from scipy.stats import chi2

from mgp_core import sample_cov


# JAX traces functions by Python function identity.  Defining both BFGS objectives
# inside ``factor_rank_tests`` used to force two fresh XLA compilations for every rank
# of every Monte Carlo replication, even though all panels have the same shapes.  Keep
# one solver pair per numerical configuration so only the first replication compiles.
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
    n = sigma.shape[0]
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
    n = sigma.shape[0]
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
    centered_x: np.ndarray, rel_jitter: float
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray], float]:
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
    # pseudoinverse or ridge repair.  The return value remains API-compatible.
    inv_omega = np.linalg.inv(omega)
    return inv_omega, tri, 0.0


def _duplication_matrix(m: int, tri: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """D with vec(A) = D vech(A) for symmetric A, in the upper-tri vech order of `tri`."""
    p = m * (m + 1) // 2
    dmat = np.zeros((m * m, p))
    for k, (i, j) in enumerate(zip(*tri)):
        dmat[i + m * j, k] = 1.0
        if i != j:
            dmat[j + m * i, k] = 1.0
    return dmat


def _factor_jacobian(loading: np.ndarray, psi: np.ndarray, m: int, r: int,
                     tri: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """G = d vech(L L' + diag(psi)) / d theta, theta = (vec L row-major, log psi)."""
    ii, jj = tri
    p = len(ii)
    g = np.zeros((p, m * r + m))
    for k in range(p):
        i, j = int(ii[k]), int(jj[k])
        for a in range(m):
            for b in range(r):
                g[k, a * r + b] = (loading[j, b] if i == a else 0.0) + (loading[i, b] if j == a else 0.0)
        if i == j:
            g[k, m * r + i] = psi[i]  # d/d(log psi_i)
    return g


def _psd_sqrt(a: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh(0.5 * (a + a.T))
    return (v * np.sqrt(np.clip(w, 0.0, None))) @ v.T


def corrected_adjusted_gof(centered_x: np.ndarray, theta_ml: np.ndarray, m: int, r: int,
                           lr_stat: float, alpha: float) -> dict[str, float]:
    """Hayakawa (2019) corrected mean--variance-adjusted goodness-of-fit test.

    Rescales the normal-theory LR statistic T_ML so it holds its size under
    non-normality AND in high dimensions: T_ML_adj = (a1/a2) T_ML ~ chi^2_{df''},
    df'' = a1^2/a2, with a1 = tr(U Omega), a2 = tr(U Omega U Omega) estimated by the
    bias-corrected Srivastava--Himeno--Yamada trace estimator, and
    U = W - W G (G'W G)^{-1} G'W the residual weight in the normal-theory metric
    W = 1/2 D'(Sigma_r^{-1} kron Sigma_r^{-1}) D. Returns a1, the corrected a2, df'',
    the adjusted statistic, and the equivalent rejection threshold on the raw T_ML scale.
    """
    nan = {"a1": np.nan, "a2c": np.nan, "dfpp": np.nan, "Jc": np.nan, "crit_corrected": np.nan}
    try:
        t = centered_x.shape[0]
        if t < m * (m + 1) // 2 + 4:
            return nan
        loading = np.asarray(theta_ml[: m * r], dtype=float).reshape(m, r)
        psi = np.exp(np.asarray(theta_ml[m * r:], dtype=float))
        sigma_r = loading @ loading.T + np.diag(psi)
        tri = np.triu_indices(m)
        dmat = _duplication_matrix(m, tri)
        sri = np.linalg.inv(sigma_r)
        w = 0.5 * dmat.T @ np.kron(sri, sri) @ dmat
        g = _factor_jacobian(loading, psi, m, r, tri)
        u = w - w @ g @ np.linalg.solve(g.T @ w @ g, g.T @ w)
        u = 0.5 * (u + u.T)
        s = (t / (t - 1.0)) * (centered_x[:, tri[0]] * centered_x[:, tri[1]])
        sc = s - s.mean(axis=0, keepdims=True)
        y = sc @ _psd_sqrt(u)
        h = y.T @ y
        yy = np.sum(y * y, axis=1)
        tr_h, tr_h2, tr_d2 = float(yy.sum()), float(np.sum(h * h)), float(np.sum(yy ** 2))
        n = t
        a1 = tr_h / (n - 1)
        a2 = ((n - 2) * (n - 1) * tr_h2 - n * (n - 1) * tr_d2 + tr_h ** 2) / (
            n * (n - 1) * (n - 2) * (n - 3)
        )
        if not (np.isfinite(a1) and np.isfinite(a2)) or a1 <= 0 or a2 <= 0:
            return nan
        dfpp = a1 * a1 / a2
        crit = float(a2 / a1 * chi2.ppf(1.0 - alpha, dfpp))
        jc = float(a1 / a2 * lr_stat) if np.isfinite(lr_stat) else np.nan
        return {"a1": float(a1), "a2c": float(a2), "dfpp": float(dfpp), "Jc": jc, "crit_corrected": crit}
    except Exception:
        return nan


def factor_rank_tests(
    x: np.ndarray,
    alpha: float = 0.05,
    min_rank: int = 1,
    max_rank: int | None = None,
    include_gmm: bool = True,
    rel_jitter: float = 1e-8,
    maxiter_ml: int = 2000,
    maxiter_gmm: int = 2000,
    tol_ml: float = 1e-6,
    tol_gmm: float = 1e-7,
    include_corrected: bool = True,
    stop_after_first_j_accept: bool = False,
    rank_search: str = "sequential",
    progress_label: str | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Old notebook factor-rank tests for a T x N grouped panel.

    For each candidate rank r, fit a Gaussian factor-analysis covariance
    Sigma_r = L L' + diag(psi), compute the likelihood-ratio statistic against
    the saturated covariance, optionally refine with the GMM objective from the
    old notebook, and compare both statistics with the chi-square rejection
    threshold. ``include_corrected=False`` skips the separate Hayakawa
    calculation when only the LR/J results are needed.  The sequential J lower
    bound is known as soon as its first acceptance is found, so
    ``stop_after_first_j_accept=True`` can shorten large-rank overview runs.
    ``rank_search='safeguarded_secant'`` treats J(r)-crit(r) as a monotone
    integer root problem, using interpolation inside a rejection/acceptance
    bracket and bisection whenever interpolation is not useful.
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
    if rank_search not in {"sequential", "safeguarded_secant"}:
        raise ValueError(f"Unknown rank_search={rank_search!r}.")

    t, n = x.shape
    if include_gmm and not j_test_reportable(n, t):
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
    if max_rank is not None:
        upper = min(upper, int(max_rank))
    lower = max(0, int(min_rank))
    if upper < lower:
        raise ValueError(f"Empty rank grid for N={n}: min_rank={lower}, max_rank={upper}.")
    ranks = list(range(lower, upper + 1))

    inv_omega = None
    tri = None
    omega_jitter = np.nan
    omega_condition = np.nan
    if include_gmm:
        inv_omega, tri, omega_jitter = _factor_rank_gmm_weight(centered, rel_jitter=rel_jitter)
        omega_condition = float(np.linalg.cond(inv_omega))

    sigma_j = jnp.asarray(sigma)
    logdet_hat_j = jnp.asarray(logdet_hat)
    inv_omega_j = jnp.asarray(inv_omega) if inv_omega is not None else None
    tri_i = jnp.asarray(tri[0]) if tri is not None else None
    tri_j = jnp.asarray(tri[1]) if tri is not None else None

    rows: list[dict[str, object]] = []
    theta_by_rank: dict[int, np.ndarray] = {}
    theta_ml_by_rank: dict[int, np.ndarray] = {}
    pending_ranks = [upper] if rank_search == "safeguarded_secant" else ranks.copy()
    gap_by_rank: dict[int, float] = {}
    while pending_ranks:
        rank = int(pending_ranks.pop(0))
        df = factor_rank_degrees_of_freedom(n, rank)
        if df <= 0:
            continue
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
        theta_gmm = theta_ml
        if include_gmm and np.isfinite(lr_stat):
            gmm_status = "ok"
            try:
                j_ml = float(gmm_obj(theta_ml, sigma_j, inv_omega_j, tri_i, tri_j, t_j))
            except Exception:
                j_ml = np.nan
            try:
                sol_gmm = gmm_solver.run(theta_ml, sigma_j, inv_omega_j, tri_i, tri_j, t_j)
                theta_gmm = sol_gmm.params
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
                theta_gmm = theta_ml
                j_stat = max(float(j_ml), 0.0) if np.isfinite(j_ml) else np.nan
                gmm_status = "ml_moment_fallback"
            elif j_stat < 0:
                j_stat = 0.0

        theta_ml_by_rank[int(rank)] = np.asarray(theta_ml, dtype=float)
        theta_by_rank[int(rank)] = np.asarray(theta_gmm if include_gmm else theta_ml, dtype=float)
        crit = float(chi2.ppf(1.0 - alpha, df))
        lr_for_p = max(lr_stat, 0.0) if np.isfinite(lr_stat) else np.nan
        j_for_p = max(j_stat, 0.0) if np.isfinite(j_stat) else np.nan
        corr = (
            corrected_adjusted_gof(centered, np.asarray(theta_ml, dtype=float), n, rank, lr_stat, alpha)
            if include_corrected and np.isfinite(lr_stat)
            else {"a1": np.nan, "a2c": np.nan, "dfpp": np.nan, "Jc": np.nan, "crit_corrected": np.nan}
        )
        crit_c = corr["crit_corrected"]
        row = {
            "r": int(rank),
            "df": int(df),
            "LR": float(lr_stat),
            "J": float(j_stat),
            "crit": crit,
            "crit_corrected": crit_c,
            "Jc": corr["Jc"],
            "dfpp": corr["dfpp"],
            "pvalue_LR": float(1.0 - chi2.cdf(lr_for_p, df)) if np.isfinite(lr_for_p) else np.nan,
            "pvalue_J": float(1.0 - chi2.cdf(j_for_p, df)) if np.isfinite(j_for_p) else np.nan,
            "reject_LR": bool(lr_stat > crit) if np.isfinite(lr_stat) else False,
            "reject_J": bool(j_stat > crit) if np.isfinite(j_stat) else False,
            # Corrected (Hayakawa 2019) test: reject when the raw LR exceeds the corrected threshold.
            "reject_corrected": bool(np.isfinite(lr_stat) and np.isfinite(crit_c) and lr_stat > crit_c),
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
        if stop_after_first_j_accept and include_gmm and np.isfinite(j_stat) and j_stat <= crit:
            break
        if rank_search == "safeguarded_secant":
            gap_by_rank[rank] = float(j_stat - crit) if np.isfinite(j_stat) else np.inf
            # Rejection at the largest overidentified rank implies rejection
            # for every smaller nested factor model.
            if rank == upper and gap_by_rank[rank] > 0:
                break
            accepted = sorted(r for r, gap in gap_by_rank.items() if gap <= 0)
            if not accepted:
                # This can only occur after a non-endpoint numerical failure;
                # retain a conservative midpoint step.
                candidate = (lower + rank) // 2
            else:
                hi = accepted[0]
                rejected = [r for r, gap in gap_by_rank.items() if r < hi and gap > 0]
                lo = max(rejected, default=lower - 1)
                if hi - lo <= 1:
                    break
                if lo >= lower and np.isfinite(gap_by_rank[lo]) and np.isfinite(gap_by_rank[hi]):
                    g_lo, g_hi = gap_by_rank[lo], gap_by_rank[hi]
                    root = lo - g_lo * (hi - lo) / (g_hi - g_lo) if g_hi != g_lo else np.nan
                    candidate = int(np.rint(root)) if np.isfinite(root) else (lo + hi) // 2
                    # Safeguard interpolation away from either endpoint.
                    candidate = min(max(candidate, lo + 1), hi - 1)
                else:
                    candidate = (lo + hi) // 2
            if candidate in gap_by_rank:
                unevaluated = [r for r in range(lower, upper + 1) if r not in gap_by_rank]
                if not unevaluated:
                    break
                candidate = min(unevaluated, key=lambda r: abs(r - candidate))
            pending_ranks.append(candidate)

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

    def lower_bound_corrected() -> float:
        # Same sequential logic on the Hayakawa-corrected test (raw LR vs corrected
        # threshold). Available only where the corrected weight is estimable.
        avail = np.isfinite(table["crit_corrected"].to_numpy(dtype=float))
        rej = table["reject_corrected"].to_numpy(dtype=bool)
        accepted = avail & (~rej)
        if accepted.any():
            return int(table.loc[accepted, "r"].iloc[0])
        if avail.any():
            return int(ledermann)
        return np.nan

    def all_rejected(stat_col: str) -> bool:
        vals = table[stat_col].to_numpy(dtype=float)
        crit_vals = table["crit"].to_numpy(dtype=float)
        return not bool((np.isfinite(vals) & (vals <= crit_vals)).any())

    r_lr = first_accepted("LR")
    r_j = first_accepted("J") if include_gmm and table["J"].notna().any() else np.nan
    has_j = bool(include_gmm and table["J"].notna().any())
    lb_lr = lower_bound("LR")
    lb_j = lower_bound("J") if has_j else np.nan
    lb_c = lower_bound_corrected()
    kbar = int(ledermann)
    selected_theta = theta_ml_by_rank.get(r_lr)
    idio_level = np.nan
    idio_share = np.nan
    if selected_theta is not None:
        psi = np.exp(selected_theta[n * r_lr:])
        idio_level = float(np.sum(psi))
        one = np.ones(n)
        agg_var = float(one @ sigma_raw @ one)
        idio_share = float(idio_level / agg_var) if agg_var > 0 else np.nan

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
        # Hayakawa (2019) small-sample-corrected test: robust to non-normality and to
        # the ADF finite-sample over-rejection that inflates factor_lb_J at small T.
        "factor_lb_corrected": int(lb_c) if pd.notna(lb_c) else np.nan,
        "regular_certified_corrected": bool(pd.notna(lb_c) and lb_c >= kbar),
        "regular_certified_J": bool(pd.notna(lb_j) and lb_j >= kbar),
        # True when no rank in the grid is accepted, so the lower bound is the
        # fallback k_bar rather than an accepted rank.
        "all_ranks_rejected_J": bool(has_j and all_rejected("J")),
        "max_testable_rank": int(table["r"].max()),
        "factor_idio_level_LR": idio_level,
        "factor_idio_share_LR": idio_share,
        "sigma_jitter": sigma_jitter,
        "omega_jitter": omega_jitter,
        "moment_cov_dimension": int(n * (n + 1) // 2),
        "moment_cov_rank": int(n * (n + 1) // 2) if include_gmm else np.nan,
        "moment_cov_condition": omega_condition,
    }
    return table, summary

#!/usr/bin/env python3
"""Monte Carlo laboratory for the inference of Section 3 (draft ``sec:inference``).

The tightness figure (``fig:gaussian_power_law_tightness``) draws one Gaussian
population per cell and reads off the *population* looseness of the grouped
ceiling. This module keeps that generative benchmark and adds the sampling
dimension, so that every claim in the inference section can be checked against a
population whose answer is known:

* power-law sizes ``S_i propto i^(-zeta)`` on ``N`` units (draft eq. for
  ``prop:powerlaw``), LPT grouping on squared sizes into ``M`` buckets;
* ``r`` Gaussian factors with loadings independent of size rank
  (Assumption ``ass:haar``), i.i.d. across units;
* heteroskedastic idiosyncratic variances -- the case the draft insists on, since
  homoskedasticity would identify the granular share outright;
* the factor scale calibrated so that the designed granular share hits a target.

Everything downstream is then exact rather than simulated:

* the truth ``gamma_0 = Gamma(Sigma_0,S)/(S'Sigma_0 S)`` from the weighted
  firm-level MGP, so the *firm-level* looseness of the MGP itself is visible;
* the estimand ``gamma_bar_0 = gamma(G Sigma_0 G')``, its population dual
  optimizer, the rank of its common part, and hence whether the population
  optimum is regular in the sense of Lemma ``lem:dirdiff``;
* the regular-case asymptotic variance in closed form,
  ``Var(u_t) = 2 tr(A Sigma A Sigma)/sigma_y^4`` with ``A = Y* - gamma_bar 11'``.

Sampling exploits the fact that the estimator only ever sees the grouped panel:
under Gaussian factors and shocks ``x_tilde_t = G x_t`` is exactly
``N(0, Sigma_tilde_0)``, so Gaussian designs are simulated in M dimensions rather
than N, and an AR(1) at the firm level is again an AR(1) after grouping. Only the
heavy-tailed design has to be drawn at the firm level, because grouping partly
Gaussianizes the fringe buckets while singleton buckets keep the raw tail -- which
is itself the interesting case.

All SDPs go through ``fast_mgp``: a replication's whole bootstrap is one batched
interior-point solve. ``run_mc_new.py`` drives the designs, ``plot_mc_new.py`` writes
the exhibits, and ``test_inference_mc.py`` checks this module against
``mgp_core``/``gradient_bootstrap``.
"""

from __future__ import annotations

import pickle
import json
from dataclasses import dataclass, field

import numpy as np

# mgp_core first: it points MOSEKLM_LICENSE_FILE at code/mosek.lic before any
# cvxpy/mosek import, and sets the matplotlib/XDG cache locations.
from mgp_core import (
    CACHE_DIR,
    Z95,
    atomic_write_pickle,
    ledermann_bound,
    lpt_buckets,
)
from near_optimal import delta_rate, near_optimal_derivative
from mosek_face import licensed_mosek
from fast_mgp import (
    common_rank,
    firm_level_mgp,
    mgp_batch,
)

RANK_REL_TOL = 1e-3          # matches mgp_core.numerical_rank / regularity_report
RANK_TIGHT_TOL = 1e-8        # near-machine tolerance, reported alongside as a sensitivity
T_SHOCK_DF = 5.0             # heavy-tailed design: finite 4th moment, as Assumption 3 needs
FIRM_MAX_ITER = 4000         # the firm-level SDP is one N x N problem, not a batch
# The certified gap 2 N mu is conservative for a weighted objective: every coordinate
# contributes mu even where S_i^2 is negligible, which at zeta = 1.4 is most of them.
# 1e-4 on Gamma_0 is still four orders below any digit the appendix reports.
FIRM_GAP_TOL = 1e-4          # accept Gamma_0 once its certified relative gap is this small
# Beside .cache/clean_panels rather than inside .cache/run_outputs, so that the
# populations -- one N-dimensional SDP each -- are retained across reruns.
POP_CACHE = CACHE_DIR / "inference_mc_populations"


# --------------------------------------------------------------------------- #
# Designs                                                                      #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Design:
    """One Monte Carlo design: a population plus a sampling scheme."""

    name: str
    label: str
    n: int = 1000                 # units
    zeta: float = 1.0             # size exponent S_i propto i^-zeta
    r: int = 3                    # common factors
    m: int = 10                   # LPT buckets
    t: int = 100                  # sample length
    rho: float = 0.0              # AR(1) coefficient at the firm level
    idio: str = "lognormal"       # "lognormal" (heteroskedastic) or "homoskedastic"
    shock: str = "gaussian"       # "gaussian" or "t5"
    idio_sd: float = 0.5          # sd of log psi_i under "lognormal"
    gamma_target: float = 0.03    # designed granular share sum S_i^2 psi_i / sigma_y^2
    delta_scale: float = 1.0      # multiplier c in delta_T = c sqrt(log T / T)
    seed: int = 20260901
    loading_draw: int = 1        # consecutive loading draw, holding idiosyncratic variances fixed

    @property
    def block_length(self) -> int:
        """Moving-block length: 8 at T=100, the draft's quarterly choice, scaled as T^(1/3)."""
        return max(2, int(round(1.75 * self.t ** (1.0 / 3.0))))

    @property
    def subsample_length(self) -> int:
        """Politis--Romano l_T = floor(T^(2/3)), as in ``mgp_core.default_subsample_length``."""
        return int(min(self.t - 1, max(3, np.floor(self.t ** (2.0 / 3.0)))))


# --------------------------------------------------------------------------- #
# Population                                                                   #
# --------------------------------------------------------------------------- #
@dataclass
class Population:
    """Exact population objects for one design. Everything here is non-random."""

    design: Design
    sizes: np.ndarray            # (N,)   size shares, sum to one
    grouping: np.ndarray         # (M,N)  G_{mi} = 1{i in B_m} S_i
    loads: np.ndarray            # (M,)   bucket squared-size loads
    psi: np.ndarray              # (N,)   idiosyncratic variances
    sigma_grouped: np.ndarray    # (M,M)  Sigma_tilde_0 = G Sigma_0 G'
    sigma_y2: float              # S' Sigma_0 S = 1' Sigma_tilde_0 1
    gamma0: float                # true granular share Gamma(Sigma_0,S)/sigma_y^2
    gamma_designed: float        # sum_i S_i^2 psi_i / sigma_y^2
    gamma_bar0: float            # grouped ceiling gamma(Sigma_tilde_0)
    ystar: np.ndarray            # (M,M)  population dual optimizer
    common: np.ndarray           # (M,M)  C* = Sigma_tilde_0 - D*
    tilde_rank: int              # rank of C* at the estimator's 1e-3 relative tolerance
    tilde_rank_tight: int        # ... and at a near-machine 1e-8 tolerance
    heywood: int                 # buckets with D*_i = 0 at the population optimum
    regular: bool                # singleton dual face, checked at population covariance
    sigma_u: float               # sd of the regular-case limit of sqrt(T)(g_hat - gamma_bar)
    half: np.ndarray = field(repr=False, default=None)   # Sigma_tilde_0^{1/2}
    grouped_loadings: np.ndarray = field(repr=False, default=None)  # (M,r) G Lambda

    @property
    def sqrt_hhi(self) -> float:
        """Square root of the Herfindahl of sizes, the concentration measure in Table 1."""
        return float(np.sqrt(np.sum(self.sizes**2)))

    @property
    def looseness(self) -> float:
        """(gamma_bar_0 - gamma_0)/gamma_0: the object the tightness section bounds."""
        return (self.gamma_bar0 - self.gamma0) / self.gamma0

    @property
    def firm_looseness(self) -> float:
        """How loose the *ungrouped* MGP already is, before any grouping."""
        return (self.gamma0 - self.gamma_designed) / self.gamma_designed


def power_law_sizes(n: int, zeta: float) -> np.ndarray:
    s = np.arange(1, n + 1, dtype=float) ** (-zeta)
    return s / s.sum()


def _psd_half(a: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh(0.5 * (a + a.T))
    return (v * np.sqrt(np.clip(w, 0.0, None))) @ v.T


def population_key(design: Design) -> tuple:
    """Fields the population depends on. T, rho and the shock law only affect sampling."""
    key = (design.n, design.zeta, design.r, design.m, design.idio,
           design.idio_sd, design.gamma_target, design.seed)
    return key if design.loading_draw == 1 else key + ("loading_draw", design.loading_draw)


def firm_key(design: Design) -> tuple:
    """Fields the *firm-level* truth depends on. The grouping M does not enter it."""
    key = (design.n, design.zeta, design.r, design.idio,
           design.idio_sd, design.gamma_target, design.seed)
    return key if design.loading_draw == 1 else key + ("loading_draw", design.loading_draw)


def feasible_objectives(sigma, diagonal, dual, weights, eps=1e-12):
    """Repair a numerical weighted-MGP solution to bound the optimum on both sides."""
    diagonal = np.maximum(np.asarray(diagonal), 0)
    floor = float(np.linalg.eigvalsh(sigma)[0])
    residual = float(np.linalg.eigvalsh(sigma - np.diag(diagonal))[0])
    diagonal *= floor / (floor + max(0., -residual) + eps)
    y = (dual + dual.T) / 2
    repair = max(0., -float(np.linalg.eigvalsh(y)[0]), float(np.max(weights - np.diag(y)))) + eps
    y += repair * np.eye(len(weights))
    return float(weights @ diagonal), float(np.sum(y * sigma))


def _firm_level_truth(design: Design, sigma_full: np.ndarray, sizes: np.ndarray,
                      use_cache: bool, objective_scale: float | None = None) -> float:
    """``Gamma(Sigma_0,S)``, cached: the one N-dimensional SDP, shared across M."""
    path = POP_CACHE / ("firm_" + _key_slug(firm_key(design)) + ".pkl")
    if use_cache and path.exists():
        with open(path, "rb") as handle:
            return float(pickle.load(handle))
    if len(sizes) >= 200:
        # The full-panel problem is solved only once per population. Use the
        # reference conic formulation; the batched barrier is for small groups.
        import cvxpy as cp
        n=len(sizes); weights=sizes**2
        scale=float(objective_scale or np.sum(weights*np.diag(sigma_full)))
        d=cp.Variable(n,nonneg=True)
        constraint=sigma_full-cp.diag(d) >> 0
        problem=cp.Problem(cp.Maximize(weights@d/scale),[constraint])
        solver="MOSEK" if licensed_mosek() else "CLARABEL"
        options={}
        if solver=="MOSEK":
            options=dict(mosek_params={"MSK_IPAR_NUM_THREADS":2,
                "MSK_DPAR_INTPNT_CO_TOL_PFEAS":1e-9,
                "MSK_DPAR_INTPNT_CO_TOL_DFEAS":1e-9,
                "MSK_DPAR_INTPNT_CO_TOL_REL_GAP":1e-9})
        problem.solve(solver=solver,**options)
        if problem.status not in ("optimal","optimal_inaccurate"):
            raise RuntimeError(f"Full-panel population SDP: {problem.status}")
        # Independently make the numerical primal and dual feasible, then
        # certify their objective gap at the same tolerance as before.
        value, upper = feasible_objectives(sigma_full, d.value, np.asarray(constraint.dual_value) * scale, weights)
        relative_gap=(upper-value)/value
        if not -1e-8 <= relative_gap <= FIRM_GAP_TOL:
            raise RuntimeError(f"Full-panel SDP primal-dual gap: {relative_gap}")
        if use_cache:
            POP_CACHE.mkdir(parents=True, exist_ok=True)
            path.with_suffix('.json').write_text(json.dumps(dict(
                method=solver,lower=value,upper=upper,relative_gap=relative_gap),indent=2)+'\n')
    else:
        firm = firm_level_mgp(sigma_full[None], sizes, max_iter=FIRM_MAX_ITER)
        if not bool(firm.converged[0]) and float(firm.relative_gap[0]) > FIRM_GAP_TOL:
            raise RuntimeError(
                f"Firm-level MGP for design {design.name!r} stalled at relative gap "
                f"{float(firm.relative_gap[0]):.2e}."
            )
        value = float(firm.value[0])
    if use_cache:
        POP_CACHE.mkdir(parents=True, exist_ok=True)
        atomic_write_pickle(value, path)
    return value


_POPULATION_CACHE: dict[tuple, "Population"] = {}


def build_population(design: Design, use_cache: bool = True) -> Population:
    """Exact population objects, cached: many designs share one population.

    The N-dimensional firm-level SDP that pins down ``gamma_0`` is the only slow
    step here (tens of seconds at N=1000), and designs that differ only in T, in
    the AR(1) coefficient or in the shock law reuse the same population.
    """
    key = population_key(design)
    if use_cache and key in _POPULATION_CACHE:
        return _replant(_POPULATION_CACHE[key], design)
    path = POP_CACHE / (_key_slug(key) + ".pkl")
    if use_cache and path.exists():
        with open(path, "rb") as handle:
            pop = pickle.load(handle)
        pop.regular = _population_regularity(pop.sigma_grouped)
        _POPULATION_CACHE[key] = pop
        return _replant(pop, design)
    pop = _build_population(design, use_cache)
    _POPULATION_CACHE[key] = pop
    if use_cache:
        POP_CACHE.mkdir(parents=True, exist_ok=True)
        atomic_write_pickle(pop, path)
    return _replant(pop, design)


# Bump when the Population layout or the numerical path changes, so stale pickles are
# rebuilt rather than unpickled into the wrong shape.
POP_CACHE_VERSION = "v2"


def _key_slug(key: tuple) -> str:
    return POP_CACHE_VERSION + "_" + "_".join(str(part) for part in key).replace(".", "p")


def _replant(pop: "Population", design: Design) -> "Population":
    """Attach the calling design (same population, possibly different T/rho/shock)."""
    if pop.design is design:
        return pop
    return Population(**{**pop.__dict__, "design": design})


def _population_regularity(sigma: np.ndarray) -> bool:
    """Use direct dual-uniqueness diagnostics, never a coarse rank proxy."""
    from mc_regularity_audit import audit_covariance
    result = audit_covariance(sigma)
    if result["regular"] is None:
        raise RuntimeError("Population dual uniqueness was numerically unresolved.")
    return bool(result["regular"])


def _build_population(design: Design, use_cache: bool = True) -> Population:
    """Exact population objects: truth, estimand, regularity and asymptotic variance."""
    rng = np.random.default_rng(design.seed)
    n, m, r = design.n, design.m, design.r
    sizes = power_law_sizes(n, design.zeta)

    if design.idio == "homoskedastic":
        psi = np.ones(n)
    elif design.idio == "lognormal":
        psi = np.exp(design.idio_sd * rng.standard_normal(n))
        psi = psi / psi.mean()
    else:
        raise ValueError(f"Unknown idio scheme {design.idio!r}.")

    # Assumption ass:haar: raw loadings i.i.d. across units, independent of size rank.
    for _ in range(design.loading_draw):
        raw_loadings = rng.standard_normal((n, r))
    granular_level = float(np.sum(sizes**2 * psi))
    factor_unit = float(np.sum((sizes @ raw_loadings) ** 2))
    if factor_unit <= 0:
        raise RuntimeError("Degenerate loading draw: aggregate factor exposure is zero.")
    # Calibrate the factor scale so sum S^2 psi / sigma_y^2 equals the target share.
    theta = granular_level * (1.0 - design.gamma_target) / (design.gamma_target * factor_unit)
    loadings = np.sqrt(theta) * raw_loadings

    buckets, loads = lpt_buckets(sizes**2, m)
    grouping = np.zeros((m, n))
    for j, bucket in enumerate(buckets):
        grouping[j, bucket] = sizes[bucket]

    # Sigma_tilde_0 = (G Lambda)(G Lambda)' + G Diag(psi) G', the second term diagonal
    # because buckets are disjoint. Never forms the N x N covariance.
    grouped_loadings = grouping @ loadings
    grouped_psi = (grouping**2) @ psi
    sigma_grouped = grouped_loadings @ grouped_loadings.T + np.diag(grouped_psi)
    sigma_y2 = float(np.ones(m) @ sigma_grouped @ np.ones(m))

    # Truth: the weighted firm-level MGP at the raw covariance (draft eq. 3).
    sigma_full = loadings @ loadings.T + np.diag(psi)
    gamma0 = _firm_level_truth(design, sigma_full, sizes, use_cache, granular_level) / sigma_y2

    grouped = mgp_batch(sigma_grouped[None])
    if not bool(grouped.converged[0]):
        raise RuntimeError(f"Grouped MGP did not converge for design {design.name!r}.")
    gamma_bar0 = float(grouped.value[0]) / sigma_y2
    ystar = grouped.dual[0]
    common = grouped.common(sigma_grouped[None])[0]
    tilde_rank = int(common_rank(common[None], RANK_REL_TOL)[0])
    tilde_rank_tight = int(common_rank(common[None], RANK_TIGHT_TOL)[0])
    heywood = int(heywood_mask(grouped.nu[0]).sum())

    # Regular-case limit: sqrt(T)(g_hat - gamma_bar) => N(0, Var(u)) with
    # u_t = (x'Y*x - gamma_bar y_t^2)/sigma_y^2; for Gaussian x, Var(x'Ax) = 2 tr(ASAS).
    a_mat = ystar - gamma_bar0 * np.ones((m, m))
    sigma_u = float(
        np.sqrt(2.0 * np.trace(a_mat @ sigma_grouped @ a_mat @ sigma_grouped)) / sigma_y2
    )

    return Population(
        design=design,
        sizes=sizes,
        grouping=grouping,
        loads=loads,
        psi=psi,
        sigma_grouped=sigma_grouped,
        sigma_y2=sigma_y2,
        gamma0=gamma0,
        gamma_designed=granular_level / sigma_y2,
        gamma_bar0=gamma_bar0,
        ystar=ystar,
        common=common,
        tilde_rank=tilde_rank,
        tilde_rank_tight=tilde_rank_tight,
        heywood=heywood,
        regular=_population_regularity(sigma_grouped),
        sigma_u=sigma_u,
        half=_psd_half(sigma_grouped),
        grouped_loadings=grouped_loadings,
    )


# --------------------------------------------------------------------------- #
# Sampling                                                                     #
# --------------------------------------------------------------------------- #
def simulate_panels(pop: Population, reps: int, rng: np.random.Generator) -> np.ndarray:
    """Draw ``reps`` grouped panels of shape (reps, T, M).

    Gaussian designs are drawn directly in the grouped space, which is exact: the
    estimator is a function of ``x_tilde_t = G x_t`` alone, and ``G x_t`` is
    Gaussian with covariance ``Sigma_tilde_0``. The AR(1) at the firm level is
    closed under grouping (``x_tilde_t = rho x_tilde_{t-1} + G eps_t``), so it is
    drawn the same way. Only the t-shock design needs the N-dimensional panel.
    """
    d = pop.design
    t, m = d.t, d.m
    if d.shock == "gaussian":
        if d.rho == 0.0:
            return rng.standard_normal((reps, t, m)) @ pop.half
        innov_half = np.sqrt(1.0 - d.rho**2) * pop.half
        out = np.empty((reps, t, m))
        state = rng.standard_normal((reps, m)) @ pop.half        # stationary start
        shocks = rng.standard_normal((reps, t, m)) @ innov_half
        for s in range(t):
            state = d.rho * state + shocks[:, s]
            out[:, s] = state
        return out
    if d.shock == "t5":
        scale = np.sqrt((T_SHOCK_DF - 2.0) / T_SHOCK_DF)
        sd = np.sqrt(pop.psi) * scale
        out = np.empty((reps, t, m))
        chunk = max(1, int(2e7 // (t * d.n)))
        for start in range(0, reps, chunk):
            k = min(chunk, reps - start)
            f = rng.standard_normal((k, t, pop.grouped_loadings.shape[1]))
            e = rng.standard_t(T_SHOCK_DF, size=(k, t, d.n)) * sd
            grouped = f @ pop.grouped_loadings.T + e @ pop.grouping.T
            if d.rho != 0.0:
                raise NotImplementedError("AR(1) with t shocks is not part of the design grid.")
            out[start : start + k] = grouped
        return out
    raise ValueError(f"Unknown shock scheme {d.shock!r}.")


def batch_cov(x: np.ndarray) -> np.ndarray:
    """Sample covariances of a stack of panels, matching ``mgp_core.sample_cov``."""
    u = x - x.mean(axis=-2, keepdims=True)
    t = x.shape[-2]
    # matmul, not einsum: einsum does not dispatch a contraction like this to BLAS and
    # runs about eight times slower on the shapes this module uses.
    return np.matmul(np.swapaxes(u, -1, -2), u) / (t - 1)


# --------------------------------------------------------------------------- #
# Point estimates                                                              #
# --------------------------------------------------------------------------- #
@dataclass
class PointEstimates:
    sigma: np.ndarray        # (K,M,M)
    share: np.ndarray        # (K,)
    se: np.ndarray           # (K,)   influence-function standard error
    nu: np.ndarray           # (K,M)  optimal diagonal D*, used for the Heywood-activity mask
    ystar: np.ndarray        # (K,M,M)
    common: np.ndarray       # (K,M,M)
    rank: np.ndarray         # (K,)  at RANK_REL_TOL, the estimator's own rule
    rank_tight: np.ndarray   # (K,)  at RANK_TIGHT_TOL
    regular: np.ndarray      # (K,) rank proxy for sample diagnostics only
    aggregate_var: np.ndarray  # (K,)  1'Sigma_hat 1


def point_estimates(x: np.ndarray, kbar: float) -> PointEstimates:
    """Vectorised replica of ``mgp_core.mgp_share`` plus the regularity diagnostic."""
    x = np.atleast_3d(np.asarray(x, dtype=float))
    k, t, m = x.shape
    sigma = batch_cov(x)
    sol = mgp_batch(sigma)
    one = np.ones(m)
    denom = sigma.sum(axis=(-2, -1))          # 1'Sigma 1
    share = sol.value / denom

    u = x - x.mean(axis=1, keepdims=True)
    quad = (np.matmul(u, sol.dual) * u).sum(axis=-1)
    agg = u @ one
    influence = (quad - share[:, None] * agg**2) / denom[:, None]
    se = influence.std(axis=1, ddof=1) / np.sqrt(t)

    common = sol.common(sigma)
    rank = common_rank(common, RANK_REL_TOL)
    return PointEstimates(
        sigma=sigma,
        share=share,
        se=se,
        nu=sol.nu,
        ystar=sol.dual,
        common=common,
        rank=rank,
        rank_tight=common_rank(common, RANK_TIGHT_TOL),
        regular=rank >= kbar,
        aggregate_var=denom,
    )


# --------------------------------------------------------------------------- #
# Resampling and confidence intervals                                          #
# --------------------------------------------------------------------------- #
# The draft's tables report the equal-tailed [5%, 95%] interval for every method,
# so the interval is nominal 90% two-sided and its upper endpoint is a nominal
# 95% one-sided upper confidence limit -- the object the applications quote.
LOW_Q, HIGH_Q = 0.05, 0.95
# Resampling draws only ever feed a quantile, so the solver runs to a looser barrier
# parameter there. The certified relative gap is then ~1e-4 -- on a share of 3% that is
# five ten-thousandths of a percentage point, five orders below anything the appendix
# reports -- and it saves a quarter of the Newton steps.
MC_MU_END = 1e-5
# Cap on the resampled-panel array held for one batched solve. B draws already give the
# solver ~300 problems per call, so there is nothing to gain from a large chunk, and a
# small one keeps several parallel workers off each other's memory bandwidth.
CHUNK_BYTES = 2.0e7

METHODS = (
    "normal",
    "bootstrap",
    "block_bootstrap",
    "gradient_bootstrap",
    "block_gradient_bootstrap",
    "subsampling",
)


@dataclass(frozen=True)
class ResampleConfig:
    n_boot: int = 299
    n_subsample: int = 300     # cap on the number of overlapping contiguous subsamples


def subsample_starts(t: int, ell: int, n_max: int) -> np.ndarray:
    """Overlapping contiguous subsample starts, thinned evenly when there are many."""
    starts = np.arange(t - ell + 1)
    if starts.size > n_max:
        starts = np.unique(np.linspace(0, starts.size - 1, n_max).round().astype(int))
    return starts


def _gather(x: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """x is (c,T,M), idx is (c,B,T) or (B,T); returns (c,B,T,M). Reference path only."""
    if idx.ndim == 2:
        return x[:, idx]
    return np.take_along_axis(x[:, None, :, :], idx[:, :, :, None], axis=2)


def _resample_counts(idx: np.ndarray, t: int) -> np.ndarray:
    """Occurrence counts of each period in each resample: (c,B,T) from (c,B,T) indices."""
    c, b, _ = idx.shape
    offsets = np.arange(c * b, dtype=np.int64)[:, None] * t
    flat = (offsets + idx.reshape(c * b, -1)).ravel()
    return np.bincount(flat, minlength=c * b * t).reshape(c, b, t).astype(float)


def _vech_index(m: int) -> tuple[np.ndarray, np.ndarray]:
    """Row/column indices of the lower triangle, and the map back to a full matrix."""
    return np.tril_indices(m)


def _outer_vech(x: np.ndarray) -> np.ndarray:
    """Lower triangle of each period's outer product: (c,T,m(m+1)/2).

    Only the triangle is carried through the resampling algebra. It is what the
    covariance needs, and it makes every product below a factor (m+1)/(2m) cheaper --
    about 1.8x at M=10 and 1.9x at M=25.
    """
    rows, cols = _vech_index(x.shape[-1])
    return x[..., rows] * x[..., cols]


def _cov_from_moments(second: np.ndarray, first: np.ndarray, n: np.ndarray, m: int) -> np.ndarray:
    """Assemble covariances from resample sums: second (…,K) in vech form, first (…,m)."""
    rows, cols = _vech_index(m)
    inv_n = 1.0 / n
    centred = second - first[..., rows] * first[..., cols] * inv_n[..., None]
    out = np.empty(centred.shape[:-1] + (m, m))
    out[..., rows, cols] = centred
    out[..., cols, rows] = centred
    return out / (n[..., None, None] - 1.0)


def cov_from_counts(x: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Sample covariances of resamples described by period counts, without materialising them.

    A resample is a multiset of periods, and the sample covariance only ever sees it
    through two sums, so there is no reason to build the (c,B,T,M) panel that carries
    it. Writing ``w`` for the count vector of one resample,

        (n-1) Sigma* = sum_t w_t x_t x_t' - (sum_t w_t x_t)(...)'/n,   n = sum_t w_t,

    both of which are linear in ``w``. Pre-multiplying the per-period outer products
    once turns the whole family of resamples into two batched matrix products against
    the count matrix. Same numbers as gathering (``test_inference_mc`` checks that to
    1e-12), roughly an order of magnitude less traffic, and BLAS instead of a strided
    gather. Contiguous resamples have more structure still: see ``cov_from_windows``.
    """
    c, t, m = x.shape
    n = counts.sum(axis=-1)
    second = counts @ _outer_vech(x)
    first = counts @ x
    return _cov_from_moments(second, first, n, m)


def window_prefix_sums(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Running sums of x and of vech(x x') over time, padded with a leading zero."""
    c, t, m = x.shape
    first = np.zeros((c, t + 1, m))
    np.cumsum(x, axis=1, out=first[:, 1:])
    outer = _outer_vech(x)
    second = np.zeros((c, t + 1, outer.shape[-1]))
    np.cumsum(outer, axis=1, out=second[:, 1:])
    return first, second


def cov_from_windows(prefix, starts: np.ndarray, lengths: np.ndarray, m: int) -> np.ndarray:
    """Covariances of resamples built from contiguous runs, from prefix sums.

    Both remaining resampling schemes are unions of contiguous runs -- a subsample is
    one run, a moving-block bootstrap draw is a handful of them -- and the sums a
    covariance needs are differences of running sums over such a run. Reading them off
    prefix sums costs one pass over T once, and O(runs) per resample thereafter, in
    place of O(T) per resample: at T = 1000 that is three hundred times less work for
    the subsamples and eighteen times less for the block bootstrap.

    ``starts`` and ``lengths`` are (…, runs); runs are summed to form one resample.
    """
    first_pref, second_pref = prefix
    c = first_pref.shape[0]
    starts, lengths = np.broadcast_arrays(starts, lengths)
    if starts.shape[0] != c:                       # runs shared by every replication
        starts = np.broadcast_to(starts, (c,) + starts.shape)
        lengths = np.broadcast_to(lengths, (c,) + lengths.shape)
    lead = np.arange(c).reshape((c,) + (1,) * (starts.ndim - 1))
    ends = starts + lengths
    first = (first_pref[lead, ends] - first_pref[lead, starts]).sum(axis=-2)
    second = (second_pref[lead, ends] - second_pref[lead, starts]).sum(axis=-2)
    n = lengths.sum(axis=-1).astype(float)
    return _cov_from_moments(second, first, n, m)


def heywood_mask(nu: np.ndarray, rel_tol: float = RANK_REL_TOL) -> np.ndarray:
    """Buckets whose recovered idiosyncratic variance is zero at the optimum.

    The dual only needs ``Y_ii >= 1`` there, so the corresponding multiplier is
    sign constrained rather than free.
    """
    return nu <= rel_tol * np.maximum(nu.max(axis=-1, keepdims=True), 1e-12)


def _block_runs(t: int, b: int, block_length: int, rng: np.random.Generator):
    """The same moving-block draw, as (starts, lengths) of contiguous runs.

    ``mgp_core.moving_block_indices`` lays ceil(T/l) blocks end to end and truncates to
    T, which shortens only the last one. Describing the draw by its runs rather than by
    T indices is what lets ``cov_from_windows`` read its moments off prefix sums.
    """
    n_blocks = int(np.ceil(t / block_length))
    starts = rng.integers(0, t - block_length + 1, size=(b, n_blocks))
    lengths = np.full((b, n_blocks), block_length)
    lengths[:, -1] = t - (n_blocks - 1) * block_length
    return starts, lengths


def _shares_from_cov(sigma: np.ndarray) -> np.ndarray:
    """Grouped shares for a stack of covariances of any leading shape."""
    flat = sigma.reshape(-1, sigma.shape[-1], sigma.shape[-1])
    sol = mgp_batch(flat, mu_end=MC_MU_END, need_dual=False)
    return (sol.value / flat.sum(axis=(-2, -1))).reshape(sigma.shape[:-2])


def _directional_shares(
    directions: np.ndarray, est: PointEstimates, rows: np.ndarray, delta: float
) -> tuple[np.ndarray, int]:
    """gamma_bar'(Sigma_hat; H) for a (C,B,M,M) stack, over the near-optimal dual set.

    The derivative is the minimum of <Y,H> over an estimate of the dual optimizer set,
    and this takes that estimate to be the set of duals that are near-optimal at the
    sample covariance,

        {Y >= 0 : diag(Y) >= 1, <Y, Sigma_hat> <= (1 + delta_T) Gamma_hat},

    rather than reading it off the rank of the recovered common part and the activity of
    its diagonal. Nothing here needs a numerical rank, a null space, or a notion of
    which regime the replication is in, and the set is bounded whenever Sigma_hat is
    positive definite, so the minimum always exists (``near_optimal``). One call carries
    every direction of every replication in the chunk.
    """
    c, b, m, _ = directions.shape
    aggregate = est.aggregate_var[rows]
    cap = est.share[rows] * aggregate * (1.0 + delta)          # (1 + delta_T) Gamma_hat
    level, solved = near_optimal_derivative(
        directions.reshape(-1, m, m), est.sigma[rows], cap,
        np.repeat(np.arange(c), b), mu_end=MC_MU_END,
    )
    trace = directions.sum(axis=(-2, -1))                       # 1'H 1
    a = aggregate[:, None]
    stat = level.reshape(c, b) / a - est.share[rows][:, None] * trace / a
    return stat, int((~solved).sum())


def _resample_covariances(x: np.ndarray, cov_fn, b: int, chunk: int) -> np.ndarray:
    """(R,B,M,M) covariances of B resamples of each panel, in chunks of replications.

    ``cov_fn(rows, c)`` returns the (c,B,M,M) covariances for one chunk. Chunking bounds
    the intermediate the scheme needs -- a count matrix, or a set of prefix sums.
    """
    reps, t, m = x.shape
    out = np.empty((reps, b, m, m))
    for lo in range(0, reps, chunk):
        rows = slice(lo, min(lo + chunk, reps))
        c = len(range(*rows.indices(reps)))
        out[rows] = cov_fn(rows, c)
    return out


def resampling_intervals(
    x: np.ndarray,
    est: PointEstimates,
    design: Design,
    cfg: ResampleConfig,
    rng: np.random.Generator,
    families: tuple[str, ...] = ("bootstrap", "block_bootstrap"),
    extra_delta_scales: tuple[float, ...] = (),
) -> dict[str, np.ndarray]:
    """All six intervals of Section ``sec:cis`` for a stack of replications.

    Returns ``{method}_low`` / ``{method}_high`` arrays plus the mean directional
    statistic, which is the draft's O(1/sqrt T) irregularity-bias estimator.
    """
    x = np.asarray(x, dtype=float)
    reps, t, m = x.shape
    b = cfg.n_boot
    rt = np.sqrt(t)
    out = {f"{name}_{side}": np.full(reps, np.nan) for name in METHODS for side in ("low", "high")}
    out["gradient_mean"] = np.empty(reps)
    out["block_gradient_mean"] = np.full(reps, np.nan)
    fallbacks = 0
    attempts = 0

    out["normal_low"] = est.share - Z95 * est.se
    out["normal_high"] = est.share + Z95 * est.se

    delta = delta_rate(t, design.delta_scale)
    ell = design.subsample_length
    starts = subsample_starts(t, ell, cfg.n_subsample)
    # The chunk bounds the count matrix and the prefix sums, not a resampled panel.
    chunk = max(1, int(CHUNK_BYTES / (max(b, len(starts)) * t * 8)))

    prefix_cache: dict[int, tuple] = {}

    def prefix_for(rows: slice):
        key = rows.start
        if key not in prefix_cache:
            prefix_cache.clear()
            prefix_cache[key] = window_prefix_sums(x[rows])
        return prefix_cache[key]

    for family in families:
        if family == "bootstrap":
            # Multinomial counts are exactly the law of T draws with replacement, and
            # skip building the index matrix at all.
            def cov_of(rows, c, rng=rng):
                return cov_from_counts(
                    x[rows], rng.multinomial(t, np.full(t, 1.0 / t), size=(c, b)).astype(float)
                )
        else:
            # Blocks are contiguous, so the moments come off prefix sums instead of a
            # pass over all T periods per draw.
            def cov_of(rows, c, rng=rng):
                runs = [_block_runs(t, b, design.block_length, rng) for _ in range(c)]
                starts = np.stack([r[0] for r in runs])
                lengths = np.stack([r[1] for r in runs])
                return cov_from_windows(prefix_for(rows), starts, lengths, m)
        sigma_b = _resample_covariances(x, cov_of, b, chunk)
        # Percentile interval, as in run_multi_source's bootstrap columns.
        shares_b = _shares_from_cov(sigma_b)
        out[f"{family}_low"] = np.quantile(shares_b, LOW_Q, axis=1)
        out[f"{family}_high"] = np.quantile(shares_b, HIGH_Q, axis=1)
        # Gradient bootstrap on the SAME draws (Fang--Santos Thm. 3.2).
        sigma_b -= est.sigma[:, None]
        sigma_b *= rt
        stat, failed = _directional_shares(sigma_b, est, np.arange(reps), delta)
        # Reuse the same covariance shocks when varying only the tolerance.
        if family == "bootstrap":
            for scale in extra_delta_scales:
                extra, extra_failed = _directional_shares(
                    sigma_b, est, np.arange(reps), delta_rate(t, scale))
                out[f"gradient_c{scale}_low"] = est.share - np.quantile(extra, HIGH_Q, axis=1) / rt
                out[f"gradient_c{scale}_high"] = est.share - np.quantile(extra, LOW_Q, axis=1) / rt
                fallbacks += extra_failed
                attempts += reps * b
        del sigma_b
        fallbacks += failed
        attempts += reps * b
        grad = "gradient_bootstrap" if family == "bootstrap" else "block_gradient_bootstrap"
        out[f"{grad}_low"] = est.share - np.quantile(stat, HIGH_Q, axis=1) / rt
        out[f"{grad}_high"] = est.share - np.quantile(stat, LOW_Q, axis=1) / rt
        key = "gradient_mean" if family == "bootstrap" else "block_gradient_mean"
        out[key] = stat.mean(axis=1)

    if cfg.n_subsample > 0:
        sigma_s = _resample_covariances(
            x,
            lambda rows, c: cov_from_windows(
                prefix_for(rows), starts[:, None], np.full((len(starts), 1), ell), m
            ),
            len(starts),
            chunk,
        )
        shares_s = _shares_from_cov(sigma_s)
        lstat = np.sqrt(ell) * (shares_s - est.share[:, None])
        out["subsampling_low"] = est.share - np.quantile(lstat, HIGH_Q, axis=1) / rt
        out["subsampling_high"] = est.share - np.quantile(lstat, LOW_Q, axis=1) / rt
    else:
        # Skipped for speed (cfg.n_subsample == 0): report the placeholder "--" rather
        # than compute it, exactly as the medium-profile empirical runs already do.
        out["subsampling_low"] = np.full(reps, np.nan)
        out["subsampling_high"] = np.full(reps, np.nan)

    out["unsolved_rate"] = np.array([fallbacks / max(attempts, 1)])
    return out


# --------------------------------------------------------------------------- #
# One design                                                                   #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class McConfig:
    reps: int = 1000
    n_boot: int = 199
    n_subsample: int = 300
    seed: int = 0


def _stream(cfg: McConfig, design: Design, tag: str) -> np.random.Generator:
    """Per-design, per-purpose RNG stream, in the spirit of ``mgp_core.make_panel_rng``."""
    digest = np.frombuffer(f"{design.name}/{tag}".encode(), dtype=np.uint8).sum()
    return np.random.default_rng([cfg.seed, int(digest), design.t, design.m])


def run_design(design: Design, cfg: McConfig,
               resampling_families: tuple[str, ...] = ("bootstrap", "block_bootstrap"),
               extra_delta_scales: tuple[float, ...] = ()) -> dict:
    """Population truth plus Monte Carlo behaviour of the estimator and the intervals."""
    pop = build_population(design)
    kbar = ledermann_bound(design.m)

    x = simulate_panels(pop, cfg.reps, _stream(cfg, design, "panels"))
    est = point_estimates(x, kbar)

    root_t = np.sqrt(design.t)
    centred = root_t * (est.share - pop.gamma_bar0)
    standardized = centred / pop.sigma_u

    row = {
        "design": design.name,
        "label": design.label,
        "loading_draw": design.loading_draw,
        "N": design.n,
        "zeta": design.zeta,
        "r": design.r,
        "M": design.m,
        "T": design.t,
        "rho": design.rho,
        "delta_scale": design.delta_scale,
        "idio": design.idio,
        "shock": design.shock,
        "reps": cfg.reps,
        "gamma0": pop.gamma0,
        "gamma_designed": pop.gamma_designed,
        "gamma_bar0": pop.gamma_bar0,
        "looseness": pop.looseness,
        "firm_looseness": pop.firm_looseness,
        "tilde_rank": pop.tilde_rank,
        "tilde_rank_tight": pop.tilde_rank_tight,
        "heywood": pop.heywood,
        "sqrt_hhi": pop.sqrt_hhi,
        "ledermann": kbar,
        "regular": pop.regular,
        "sigma_u": pop.sigma_u,
        "cv_loads": float(pop.loads.std(ddof=0) / pop.loads.mean()),
        "mean_share": float(est.share.mean()),
        "bias": float(est.share.mean() - pop.gamma_bar0),
        "bias_se": float(est.share.std(ddof=1) / np.sqrt(cfg.reps)),
        "sd_share": float(est.share.std(ddof=1)),
        "mean_se": float(est.se.mean()),
        "se_ratio": float(est.se.mean() / est.share.std(ddof=1)),
        "skew": float(_moment(standardized, 3)),
        "excess_kurtosis": float(_moment(standardized, 4) - 3.0),
        "sd_standardized": float(standardized.std(ddof=1)),
        "sample_regular_share": float(est.regular.mean()),
        "sample_regular_share_tight": float((est.rank_tight >= kbar).mean()),
        "mean_sample_rank": float(est.rank.mean()),
        "mean_sample_rank_tight": float(est.rank_tight.mean()),
        "mean_sample_heywood": float(heywood_mask(est.nu).sum(axis=1).mean()),
    }
    intervals = resampling_intervals(
        x, est, design, ResampleConfig(cfg.n_boot, cfg.n_subsample),
        _stream(cfg, design, "resample"),
        families=resampling_families, extra_delta_scales=extra_delta_scales,
    )
    row["reps_resampling"] = cfg.reps
    row["bias_irregularity_hat"] = float(intervals["gradient_mean"].mean()) / root_t
    row["unsolved_rate"] = float(intervals["unsolved_rate"][0])
    coverage_rows = _coverage_table(design, pop, intervals)
    tolerance_coverage = {}
    for scale in extra_delta_scales:
        varied = dict(intervals)
        varied["gradient_bootstrap_low"] = intervals[f"gradient_c{scale}_low"]
        varied["gradient_bootstrap_high"] = intervals[f"gradient_c{scale}_high"]
        tolerance_coverage[scale] = [item for item in _coverage_table(design, pop, varied)
                                   if item["method"] == "gradient_bootstrap"]
    return {
        "population": pop,
        "row": row,
        "coverage": coverage_rows,
        "tolerance_coverage": tolerance_coverage,
        "standardized": standardized,
        "share": est.share,
        "se": est.se,
    }


def _moment(z: np.ndarray, order: int) -> float:
    zc = z - z.mean()
    return float(np.mean(zc**order) / np.mean(zc**2) ** (order / 2.0))


def _coverage_table(
    design: Design,
    pop: Population,
    intervals: dict[str, np.ndarray],
) -> list[dict]:
    """Two-sided and one-sided coverage of both the estimand and the truth."""
    rows = []
    for name in METHODS:
        lo, hi = intervals[f"{name}_low"], intervals[f"{name}_high"]
        # A skipped construction (e.g. subsampling with cfg.n_subsample == 0) reports
        # all-NaN bounds; propagate NaN rather than let "NaN <= x" silently read as a
        # real zero-coverage result.
        missing = bool(np.all(np.isnan(hi)))
        nan = float("nan")
        rows.append(
            {
                "design": design.name,
                "label": design.label,
                "method": name,
                "reps": len(lo),
                "cover_two_sided_estimand": nan if missing else float(np.mean((lo <= pop.gamma_bar0) & (pop.gamma_bar0 <= hi))),
                "cover_upper_estimand": nan if missing else float(np.mean(pop.gamma_bar0 <= hi)),
                "cover_upper_truth": nan if missing else float(np.mean(pop.gamma0 <= hi)),
                # The planted factor-model share, the third population object: an
                # interval can cover the estimand and the truth at very different
                # rates once the grouped ceiling is loose.
                "cover_upper_designed": nan if missing else float(np.mean(pop.gamma_designed <= hi)),
                "cover_two_sided_truth": nan if missing else float(np.mean((lo <= pop.gamma0) & (pop.gamma0 <= hi))),
                "median_upper": nan if missing else float(np.median(hi)),
                "median_length": nan if missing else float(np.median(hi - lo)),
            }
        )
    return rows
